from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import time
from datetime import datetime, timedelta

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource, Tone
from ..actions.query_processing import get_search_results

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
    def __init__(self, researcher):
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
        self.context = []  # Track all context

    async def generate_search_queries(self, query: str, search_plan: Dict[str, str], current_search_step: str, num_queries: int = 3) -> List[Dict[str, str]]:
        """Generate SERP queries for research"""
        messages = [
            {"role": "system", "content": "You are an expert researcher generating search queries."},
            {"role": "user",
             "content": f"""Given the following question and the search plan, generate {num_queries} unique search queries to explore the step {current_search_step} thoroughly. 
             
             For each query, provide a research goal. 
             Format as 'Query: <query>' followed by 'Goal: <goal>' for each pair. 

             Here is the question and the entire search plan: {query}"""}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.Medium.value,
            temperature=0.4
        )

        lines = response.split('\n')
        queries = []
        current_query = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Query:'):
                if current_query:
                    queries.append(current_query)
                current_query = {'query': line.replace('Query:', '').strip()}
            elif line.startswith('Goal:') and current_query:
                current_query['researchGoal'] = line.replace('Goal:', '').strip()

        if current_query:
            queries.append(current_query)

        return queries[:num_queries]

    async def generate_research_plan(self, query: str, num_questions: int = 3) -> List[str]:
        """Generate follow-up questions to clarify research direction"""
        # Get initial search results to inform query generation
#         search_results = await get_search_results(query, self.researcher.retrievers[0])
#         logger.info(f"Initial web knowledge obtained: {len(search_results)} results")

#         # Get current time for context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         messages = [
#             {"role": "system", "content": "You are an expert researcher. Your task is to analyze the original query and search results, then generate targeted questions that explore different aspects and time periods of the topic."},
#             {"role": "user",
#              "content": f"""Original query: {query}

# Current time: {current_time}

# Search results:
# {search_results}

# Based on these results, the original query, and the current time, generate {num_questions} unique questions. Each question should explore a different aspect or time period of the topic, considering recent developments up to {current_time}.

# Format each question on a new line starting with 'Question: '"""}
#         ]
#         messages = [
#             {"role": "system", "content": '''You are a query analyst. Your task is to analyze the user's question and identify their underlying intent — whether they expect you to: provide a direct answer, write a report, solve a math problem, etc.Use the following format:\nUser: "{user's question}"\n Assistant: "{the expected task}"\n
# Examples: 
# User: "What is the capital of China?"
# Assistant: "Provide a direct answer to the question."
             
# User: "Can you analyze Zhu Yuanzhang's personality?"
# Assistant: "Write a report analyzing Zhu Yuanzhang's personality."
#             '''},

#             {"role": "user",
#              "content": f"""{query}"""}
#         ]
        
#         task_response = await create_chat_completion(
#             messages=messages,
#             llm_provider=self.researcher.cfg.strategic_llm_provider,
#             model=self.researcher.cfg.strategic_llm_model,
#             reasoning_effort=ReasoningEfforts.High.value,
#             temperature=0.4
#         )
        # task = task_response.split("Assistant: ")[1].strip()
        # print(task)
        # messages = [
        #     {"role": "system", "content": f"""You are an expert researcher. Your task is to solve the user's request, it could be an answer to a question, a report, a math proof, etc."""},
        #     {"role": "user",
        #      "content": f"""Original query: {query}.  
             
        #      Let's first understand the problem and devise a search plan to solve the problem. 
             
        #      Format requirement:
        #      - Format each plan step on a new line starting with 'Plan Step A: ', 'Plan Step B: ', etc.
        #      - To start with, you need to say 'My plan is: '"""}
        # ]
        messages = [
            {"role": "system", "content": f"""You are a highly capable research assistant with expertise in breaking down complex problems and devising effective strategies for information retrieval and reasoning. Your role is to help address the user's request, which may involve answering a question, generating a report, or constructing a logical or mathematical proof."""},
            
            {"role": "user", "content": f"""Original query: {query}

        Your task is to first analyze and interpret the user's query to understand its underlying objectives, constraints, and required knowledge domains. Then, based on your understanding, formulate a structured search or research plan that could be executed to solve the problem effectively.

        **Output formatting requirements:**
        - Begin your response with: `My search plan is:`
        - Present each step of the plan on a new line, prefixed sequentially as: `Plan Step 1:`, `Plan Step 2:`, and so on.
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

    async def process_research_results(self, query: str, context: str, num_learnings: int = 3) -> Dict[str, List[str]]:
        """Process research results to extract learnings and follow-up questions"""
        messages = [
            {"role": "system", "content": "You are an expert analyzing search results."},
            {"role": "user",
             "content": f"""Given the following search results for the query '{query}', extract key learnings from the results step by step.

             **Output formatting requirements:**
             - For each learning, include a citation to the source URL if available. 
             - Format each learning as 'Learning [source_url]: <insight>'.
             - Change each learning to a new line.
             \n\n
             Here is the search results: {context}"""
             }
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.4,
            reasoning_effort=ReasoningEfforts.High.value,
            max_tokens=1000
        )

        lines = response.split('\n')
        learnings = []
        # questions = []
        citations = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Learning'):
                import re
                url_match = re.search(r'\[(.*?)\]:', line)
                if url_match:
                    url = url_match.group(1)
                    learning = line.split(':', 1)[1].strip()
                    learnings.append(learning)
                    citations[learning] = url
                else:
                    # Try to find URL in the line itself
                    url_match = re.search(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
                    if url_match:
                        url = url_match.group(0)
                        learning = line.replace(url, '').replace('Learning:', '').strip()
                        learnings.append(learning)
                        citations[learning] = url
                    else:
                        learnings.append(line.replace('Learning:', '').strip())
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
            breadth: int,
            search_plan: Dict[str, str],
            current_search_step: str,
            learnings: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
            on_progress=None
    ) -> Dict[str, Any]:
        """Conduct deep iterative research"""
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()

        progress = ResearchProgress(depth, breadth)

        if on_progress:
            on_progress(progress)

        depth = len(search_plan.keys()) - current_search_step
        # Generate search queries
        serp_queries = await self.generate_search_queries(query, search_plan, current_search_step, num_queries=breadth)
        f = open('queries.txt', 'a')
        f.write("Depth: " + str(depth) + "; Breadth: " + str(breadth) + "\n" + "Query: " + query + "\n" + " SERP Queries: " + str(serp_queries) + "\n\n")
        f.close()
        progress.total_queries = len(serp_queries)
        all_learnings = learnings.copy()
        all_citations = citations.copy()
        all_visited_urls = visited_urls.copy()
        all_context = []
        all_sources = []

        # Process queries with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_query(serp_query: Dict[str, str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    progress.current_query = serp_query['query']
                    if on_progress:
                        on_progress(progress)

                    from .. import GPTResearcher
                    researcher = GPTResearcher(
                        query=serp_query['query'],
                        report_type=ReportType.ResearchReport.value,
                        report_source=ReportSource.Web.value,
                        tone=self.tone,
                        websocket=self.websocket,
                        config_path=self.config_path,
                        headers=self.headers,
                        visited_urls=self.visited_urls
                    )

                    # Conduct research
                    context = await researcher.conduct_research()

                    # Get results and visited URLs
                    visited = researcher.visited_urls
                    sources = researcher.research_sources

                    # Process results to extract learnings and citations
                    results = await self.process_research_results(
                        query=serp_query['query'],
                        context=context
                    )

                    # Update progress
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                    if on_progress:
                        on_progress(progress)

                    return {
                        'learnings': results['learnings'],
                        'visited_urls': list(visited),
                        # 'followUpQuestions': results['followUpQuestions'],
                        'researchGoal': serp_query['researchGoal'],
                        'citations': results['citations'],
                        'context': context if context else "",
                        'sources': sources if sources else []
                    }

                except Exception as e:
                    logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                    return None

        # Process queries concurrently with limit
        tasks = [process_query(query) for query in serp_queries]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Update breadth progress based on successful queries
        progress.current_breadth = len(results)
        if on_progress:
            on_progress(progress)

        # Collect all results
        for result in results:
            all_learnings.extend(result['learnings'])
            all_visited_urls.update(result['visited_urls'])
            all_citations.update(result['citations'])
            if result['context']:
                all_context.append(result['context'])
            if result['sources']:
                all_sources.extend(result['sources'])

            # Continue deeper if needed
        if depth > 1:
            new_breadth = max(2, breadth // 2)
            # new_depth = depth - 1
            progress.current_depth += 1

            # Create next query from research goal and follow-up questions
            # next_query = f"""
            # Previous research goal: {result['researchGoal']}
            # Follow-up questions: {' '.join(result['followUpQuestions'])}
            # """

            # Recursive research
            deeper_results = await self.deep_research(
                query=query,
                breadth=new_breadth,
                search_plan=search_plan,
                current_search_step=list(search_plan.keys())[int(current_search_step) + 1],
                learnings=all_learnings,
                citations=all_citations,
                visited_urls=all_visited_urls,
                on_progress=on_progress
            )

            all_learnings = deeper_results['learnings']
            all_visited_urls.update(deeper_results['visited_urls'])
            all_citations.update(deeper_results['citations'])
            if deeper_results.get('context'):
                all_context.extend(deeper_results['context'])
            if deeper_results.get('sources'):
                all_sources.extend(deeper_results['sources'])

        # Update class tracking
        self.context.extend(all_context)
        self.research_sources.extend(all_sources)

        # Trim context to stay within word limits
        trimmed_context = trim_context_to_word_limit(all_context)
        logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

        return {
            'learnings': list(set(all_learnings)),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'context': trimmed_context,
            'sources': all_sources
        }

    async def run(self, on_progress=None) -> str:
        """Run the deep research process and generate final report"""
        start_time = time.time()

        # Log initial costs
        initial_costs = self.researcher.get_costs()

        plan_steps, plan_steps_numbers, plan_steps_contents = await self.generate_research_plan(self.researcher.query)
        # answers = ["Automatically proceeding with research"] * len(follow_up_questions)

        # qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)]
        # combined_query = f"""
        # Initial Query: {self.researcher.query}\nFollow - up Questions and Answers:\n
        # """ + "\n".join(qa_pairs)
        combined_query = f"""
        Initial Query: {self.researcher.query}\nThe search plan is:\n
        """ + "\n".join([f"{n}: {c}" for n, c in zip(plan_steps_numbers, plan_steps_contents)])

        search_plan = {n: c for n, c in zip(plan_steps_numbers, plan_steps_contents)}
        results = await self.deep_research(
            query=combined_query,
            search_plan=search_plan,
            current_search_step=list(search_plan.keys())[0],
            breadth=self.breadth,
            on_progress=on_progress
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

        # Add all research context
        if results.get('context'):
            context_with_citations.extend(results['context'])

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