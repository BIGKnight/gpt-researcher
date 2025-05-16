from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import time
from datetime import datetime, timedelta
import re
import os
# import numpy as np

from synvo_researcher.gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource, Tone
from ..actions.query_processing import get_search_results
from synvofs.utils.sse.sse_manager import SSEQueue
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Maximum words allowed in context (25k words for safety margin)
MAX_CONTEXT_WORDS = 250000

def count_words(text: str) -> int:
    """Count words in a text string"""
    return len(text.split())

def trim_context_to_word_limit(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """Trim context list to stay within word limit while preserving most recent/relevant items"""
    total_words = 0
    trimmed_context = []

    # Process in reverse to keep most recent items
    for item in reversed(context_list):
        words = count_words(item)
        if total_words + words <= max_words:
            trimmed_context.insert(0, item)  # Insert at start to maintain original order
            total_words += words
        else:
            break

    return trimmed_context

class ResearchProgress:
    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth = 1  # Start from 1 and increment up to total_depth
        self.total_depth = total_depth
        self.current_breadth = 0  # Start from 0 and count up to total_breadth as queries complete
        self.total_breadth = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries = 0
        self.completed_queries = 0


class DeepResearchSkill:
    def __init__(self, researcher, sse_queue: SSEQueue | None = None):
        self.researcher = researcher
        self.breadth = getattr(researcher.cfg, 'deep_research_breadth', 4)
        self.depth = getattr(researcher.cfg, 'deep_research_depth', 2)
        self.concurrency_limit = getattr(researcher.cfg, 'deep_research_concurrency', 2)
        self.websocket = researcher.websocket
        self.tone = researcher.tone
        self.config_path = researcher.cfg.config_path if hasattr(researcher.cfg, 'config_path') else None
        self.headers = researcher.headers or {}
        self.visited_urls = researcher.visited_urls
        self.learnings = []
        self.research_sources = []  # Track all research sources
        # self.context = []  # Track all context
        self.sse_queue = sse_queue
        if not self.sse_queue:
            self.sse_queue = SSEQueue()

    async def generate_search_queries(self, query: str, previous_learnings: str, current_search_target: str) -> List[Dict[str, str]]:
        """Generate SERP queries for research"""
        # string_conclusions = "\n".join(intermediate_conclusions) if len(intermediate_conclusions) > 0 else "No previous conclusions."
        messages = [
            {"role": "system", "content": "You are an expert researcher generating search queries."},
            {"role": "user",
             "content": f"""Given the following question and the search objective, generate several search queries to explore it thoroughly. 

             **Output Requirements:**
             - For each query, you need to provide a step-by-step thought process for why you are searching it, aling with a brief reason as summary of the thought process.
             - Each query should explore a different aspect of the question.
             - Format as '<thought Process>\nQuery: <query>\nReason: <reason>\n\n<thought Process>\nQuery: <query>\nReason: <reason>\n ...'. 
             - For each query, you should limit the number of words to 10.
             - The number of queries should less than 5. 

             Here is the question: {query} \n\n
             Here is the current search objective: {current_search_target}\n\n
             Here are some previous learnings: {previous_learnings}\n\n
"""}
        ]
        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.Medium.value,
            temperature=0.4
        )
        os.makedirs('deep_research_output', exist_ok=True)
        f = open('deep_research_output/search_queries_response.txt', 'a')
        f.write("generate_search_queries_response: \n" + response + "\n")
        f.close()
        lines = response.split('\n')
        queries = []
        current_query = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Query:'):
                if current_query:
                    queries.append(current_query)
                current_query = {'query': line.replace('Query:', '').strip()}
            elif line.startswith('Reason:') and current_query:
                current_query['researchGoal'] = line.replace('Reason:', '').strip()

        if current_query:
            queries.append(current_query)

        return queries

    async def generate_research_plan(self, query: str) -> List[str]:
        messages = [
            {"role": "system", "content": f"""You are a highly capable research assistant with expertise in breaking down complex problems and devising effective strategies for information retrieval and reasoning. Your role is to help address the user's request, which may involve answering a question, generating a report, or constructing a logical or mathematical proof."""},
            
            {"role": "user", "content": f"""Original query: {query}

        Your task is to first analyze and interpret the user's query to understand its underlying objectives, constraints, and required knowledge domains. Then, based on your understanding, formulate a structured search or research plan that could be executed to solve the problem effectively.

        **Output formatting requirements:**
        - Begin your response with: `My search plan is:`
        - Present each step of the plan on a new line, prefixed sequentially as: `Plan Step 1:`, `Plan Step 2:`, and so on.
        - The total number of steps should be less or equal to 6.
        - Ensure that each step is actionable, logically ordered, and contributes meaningfully to solving the original query."""}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            max_tokens=10000,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.High.value,
            temperature=0.4
        )
        plan = response.replace('My search plan is:', '').strip()
        plan_steps = [q.replace('Plan Step ', '').strip() for q in response.split('\n') if q.strip().startswith('Plan Step ')]
        plan_steps_numbers = [q.split(':')[0].strip() for q in plan_steps]
        plan_steps_contents = [q.split(':')[1].strip() for q in plan_steps]
        return plan_steps, plan_steps_numbers, plan_steps_contents
        # questions = [q.replace('Question:', '').strip()
        #              for q in response.split('\n')
        #              if q.strip().startswith('Question:')]
        # return questions[:num_questions]

    async def process_research_results(self, query: str, context: str, previous_learnings: str, num_learnings: int = 3) -> Dict[str, List[str]]:
        """Process research results to extract learnings and follow-up questions"""
        messages = [
            {"role": "system", "content": "You are an expert analyzing search results."},
            {"role": "user",
             "content": f"""Given the following search results and some previous learnings for the query '{query}', extract key clues.

             **Output Requirements:**
             - For each clue, you need to provide a step-by-step thought process for why you conclude it.
             - For each clue, include a citation to the source URL if available. 
             - Format as '<thought Process>\nClue [source_url]: <insight>\n<thought Process>\nClue [source_url]: <insight>\n ...'.
             \n\n
             Here is the search results: 
             {context}
             \n\n
             Here are some previous learnings:
             {previous_learnings}
"""
             }
        ]

        async with self.sse_queue.operation(f"🧠 Processing research results for query: {query}") as operation:
            response = await create_chat_completion(
                messages=messages,
                llm_provider=self.researcher.cfg.strategic_llm_provider,
                model=self.researcher.cfg.strategic_llm_model,
                temperature=0.4,
                reasoning_effort=ReasoningEfforts.High.value,
                max_tokens=1000
            )

            os.makedirs('deep_research_output', exist_ok=True)
            f = open('deep_research_output/process_research_results_response.txt', 'a')
            f.write("process_research_results_response: \n" + response + "\n")
            f.close()
            lines = response.split('\n')
            learnings = []
            # questions = []
            
            citations = {}

            for line in lines:
                line = line.strip()
                if line.startswith('Clue'):
                    import re
                    url_match = re.search(r'\[(.*?)\]:', line)
                    if url_match:
                        url = url_match.group(1)
                        learning = line.split(':', 1)[1].strip()
                        learnings.append(learning)
                        citations[learning] = url
                        await operation.output_url(url, start_text="- Found useful URL: 🔗 ", end_text="\n")
                    else:
                        # Try to find URL in the line itself
                        url_match = re.search(
                            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
                        if url_match:
                            url = url_match.group(0)
                            learning = line.replace(url, '').replace('Clue:', '').strip()
                            learnings.append(learning)
                            citations[learning] = url
                            await operation.output_url(url, start_text="Found useful URL: 🔗 ")
                        else:
                            learnings.append(line.lstrip('Clue:').strip())
                            await operation.output(f"Found clue: {line.lstrip('Clue:').strip()}", end="\n")
            # elif line.startswith('Question:'):
            #     questions.append(line.replace('Question:', '').strip())

        return {
            'learnings': learnings[:num_learnings],
            # 'followUpQuestions': questions[:num_learnings],
            'citations': citations
        }

    async def deep_research(
            self,
            query: str,
            search_plan: Dict[str, str],
            # initial_search_step: str,
            learnings: List[str] = None,
            # intermediate_conclusions: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
    ) -> Dict[str, Any]:
        """Conduct deep iterative research"""
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()
        # if intermediate_conclusions is None:
        #     intermediate_conclusions = []

        all_learnings = learnings.copy()
        all_citations = citations.copy()
        all_visited_urls = visited_urls.copy()
        # all_conclusions = intermediate_conclusions.copy()
        all_context = []
        all_sources = []
        # cur_search_plan = [initial_search_step]
        # current_search_step = list(search_plan.keys())[0]
        
        count = 0
        for search_step, search_goal in search_plan.items():
            if len(all_learnings) > 0:
                previous_learnings="\n".join(all_learnings)
            else:
                previous_learnings="No previous learnings."
            async with self.sse_queue.operation(f"✨ Generating search queries") as operation:
                serp_queries = await self.generate_search_queries(query, previous_learnings, current_search_target=search_goal)
                table = [{"No.": i + 1, "Query": q['query'], "Research Goal": q['researchGoal']} for i, q in enumerate(serp_queries)]
                output_table = tabulate(table, headers="keys", tablefmt="github")
                logger.info(output_table)
                await operation.output(output_table)
            os.makedirs('deep_research_output', exist_ok=True)
            f = open('deep_research_output/queries.txt', 'a')
            f.write("current_search_step: " + str(count + 1) + "\n" + "query: " + query + "\n" + " SERP Queries: " + str(serp_queries) + "\n\n")
            f.write("search_plan: \n" + "\n".join([f"Plan Step {k}: {v}" for k, v in search_plan.items()]) + "\n")
            # if count > 0:
            #     f.write("conclusions: \n" + "\n".join([f"conclusion: {c}" for c in all_conclusions]) + "\n\n")
            f.close()

            # Process queries with concurrency limit
            semaphore = asyncio.Semaphore(self.concurrency_limit)

            async def process_query(serp_query: Dict[str, str], previous_learnings: str) -> Optional[Dict[str, Any]]:
                async with semaphore:
                    try:
                        from .. import GPTResearcher
                        researcher = GPTResearcher(
                            query=serp_query['query'],
                            report_type=ReportType.ResearchReport.value,
                            report_source=ReportSource.Web.value,
                            tone=self.tone,
                            websocket=self.websocket,
                            config_path=self.config_path,
                            headers=self.headers,
                            visited_urls=self.visited_urls,
                            sse_queue=self.sse_queue
                        )
                        # Conduct research
                        context = await researcher.conduct_research()
                        # Get results and visited URLs
                        visited = researcher.visited_urls
                        sources = researcher.research_sources
                        # Process results to extract learnings and citations
                        results = await self.process_research_results(
                            query=serp_query['query'],
                            previous_learnings=previous_learnings,
                            context=context
                        )
                        return {
                            'learnings': results['learnings'],
                            'visited_urls': list(visited),
                            'researchGoal': serp_query['researchGoal'],
                            'citations': results['citations'],
                            'context': context if context else "",
                            'sources': sources if sources else []
                        }

                    except Exception as e:
                        logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                        return None

            # Process queries concurrently with limit
            tasks = [process_query(query, "\n".join(all_learnings)) for query in serp_queries]
            results = await asyncio.gather(*tasks)
            results = [r for r in results if r is not None]

            # Collect all results
            for result in results:
                all_learnings.extend(result['learnings'])
                all_visited_urls.update(result['visited_urls'])
                all_citations.update(result['citations'])
                if result['context']:
                    all_context.append(result['context'])
                if result['sources']:
                    all_sources.extend(result['sources'])

            # self.context.extend(all_context)
            # Trim context to stay within word limits
            self.research_sources.extend(all_sources)
            trimmed_context = trim_context_to_word_limit(all_context)
            logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

            # summarize the learnings
            # string_intermediate_conclusions = "\n".join(intermediate_conclusions)
            # string_current_learnings = "\n".join(result['learnings'])
    #         messages_conclude = [
    #             {"role": "system", "content": "You are an expert analyst. "},
    #             {"role": "user", "content": f"""Based on previous conclusions and the current learnings, extract a key conclusion. It is not allowed to say "I don't know", "I don't have any information", etc.

    # **Format:**: "conclusion: <conclusion>."

    # Here are the previous conclusions: \n
    # {string_intermediate_conclusions} \n

    # Here are the current learnings: \n
    # {string_current_learnings}"""
    # }
    #         ]
    #         response_conclude = await create_chat_completion(
    #             messages=messages_conclude,
    #             llm_provider=self.researcher.cfg.strategic_llm_provider,
    #             model=self.researcher.cfg.strategic_llm_model,
    #             temperature=0.4,
    #             reasoning_effort=ReasoningEfforts.High.value,
    #             max_tokens=1000
    #         )

    #         conclusion = response_conclude.split("conclusion:")[1].strip()
    #         all_conclusions.append(conclusion)

    #         string_all_conclusions = "\n".join(all_conclusions)
    #         messages_revise = [
    #             {"role": "system", "content": "You are an expert summarizer."},
    #             {"role": "user", "content": f"""Based on the current conclusions, propose the subsequent search plan step.
                 
    # **Format:**: "Next search step is: <step_content>. \n "

    # Here are the conclusions: \n
    # {string_all_conclusions}"""
    # }
    #         ]

    #         response_revise = await create_chat_completion(
    #             messages=messages_revise,
    #             llm_provider=self.researcher.cfg.strategic_llm_provider,
    #             model=self.researcher.cfg.strategic_llm_model,
    #             temperature=0.4,
    #             reasoning_effort=ReasoningEfforts.High.value,
    #             max_tokens=1000
    #         )

    #         new_plan = response_revise.replace('Next search step is: ', '').strip()
    #         cur_search_plan.append(new_plan)
            count += 1

        return {
            'learnings': list(set(all_learnings)),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'context': trimmed_context,
            'sources': all_sources
        }

    async def run(self) -> str:
        """Run the deep research process and generate final report"""
        start_time = time.time()

        # Log initial costs
        initial_costs = self.researcher.get_costs()

        plan_steps, plan_steps_numbers, plan_steps_contents = await self.generate_research_plan(self.researcher.query)
        # combined_query = f"""
        # Initial Query: {self.researcher.query}\nThe search plan is:\n
        # """ + "\n".join([f"{n}: {c}" for n, c in zip(plan_steps_numbers, plan_steps_contents)])

        search_plan = {n: c for n, c in zip(plan_steps_numbers, plan_steps_contents)}
        results = await self.deep_research(
            query=self.researcher.query,
            search_plan=search_plan
        )

        # Get costs after deep research
        research_costs = self.researcher.get_costs() - initial_costs

        # Log research costs if we have a log handler
        if self.researcher.log_handler:
            await self.researcher._log_event("research", step="deep_research_costs", details={
                "research_costs": research_costs,
                "total_costs": self.researcher.get_costs()
            })

        # Prepare context with citations
        context_with_citations = []
        for learning in results['learnings']:
            citation = results['citations'].get(learning, '')
            if citation:
                context_with_citations.append(f"{learning} [Source: {citation}]")
            else:
                context_with_citations.append(learning)

        # for conclusion in results['all_conclusions']:
        #     context_with_citations.append(f"Intermediate clue: {conclusion}")

        # Add all research context
        # if results.get('context'):
        #     context_with_citations.extend(results['context'])

        # Trim final context to word limit
        final_context = trim_context_to_word_limit(context_with_citations)
        
        # Set enhanced context and visited URLs
        self.researcher.context = "\n".join(final_context)
        self.researcher.visited_urls = results['visited_urls']

        # Set research sources
        if results.get('sources'):
            self.researcher.research_sources = results['sources']

        # Log total execution time
        end_time = time.time()
        execution_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Total research execution time: {execution_time}")
        logger.info(f"Total research costs: ${research_costs:.2f}")

        # Return the context - don't generate report here as it will be done by the main agent
        return self.researcher.context