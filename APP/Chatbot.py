from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
import re
from hybrid_retrieval import hybrid_retrieval
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Router(BaseModel):
    mode: Literal["pdf", "summary", "general"] = Field(
        description=(
            "pdf = needs specific facts/sections from one or more uploaded PDFs; "
            "summary = needs a whole-document overview rather than a specific chunk; "
            "general = no retrieval needed (greetings, casual talk, general knowledge)."
        )
    )
    retrieve_query: Optional[str] = Field(
        default=None,
        description="Rewritten standalone retrieval query. Null if mode is 'general'."
    )


router_model = model.with_structured_output(Router)
MAX_HISTORY_MESSAGES  = 16

ROUTER_SYSTEM_PROMPT = """
Pick exactly one mode:
- general: greetings, casual talk, questions needing no PDF context
- pdf: needs specific facts/sections from the uploaded PDF(s)
- summary: needs the whole document's overview

If mode is 'pdf', rewrite the query into a standalone retrieval query.
                        """

def _build_source_filter(active_sources: list[str] | None):

    if not active_sources:
        return None
    if len(active_sources) == 1:
        return {"source": active_sources[0]}

    return {"source": {"$in": active_sources}}


def _extract_used_citations(answer_text: str , source_map : dict) -> list[dict]:
    """
        Scans the model's answer for [n] markers, keeps only the ones that
        correspond to a real entry in source_map (guards against the model
        inventing a citation number that was never provided), and returns
        them in first-appearance order with duplicates removed.
    """
    found_numbers = re.findall(r"\[(\d+)\]", answer_text)
    used = []
    seen = set()
    for n in found_numbers:
        n = int(n)
        if n in source_map and n not in seen:
            used.append({"citation_id": n , **source_map[n]})
            seen.add(n)
    return used

def _trim_history(history: list , max_messages: int = MAX_HISTORY_MESSAGES) ->list:
    trimmed = history
    while len(trimmed) > max_messages:
        trimmed = trimmed[2:]
    return trimmed


class ChatEngine:

    def __init__(self,chunks_store,summary_store):
        self.chunks_store = chunks_store
        self.summary_store = summary_store
        self.chat_history = []

    def ask(self,query:str,active_sources:list[str] | None = None) ->dict:

        self.chat_history.append(HumanMessage(content = query))
        self.chat_history = _trim_history(self.chat_history)

        routing_messages = [
            SystemMessage(content = ROUTER_SYSTEM_PROMPT),
            *self.chat_history,
        ]

        decision = router_model.invoke(routing_messages)

        source_filter = _build_source_filter(active_sources)
        cited_sources = []
        pdf_source_map = None

        if decision.mode == "pdf":
            search_query = decision.retrieve_query or query
            results =  hybrid_retrieval(self.chunks_store,search_query,source_filter=source_filter , k =4)

            if not results:
                system_prompt = (
                "No relevant PDF content was found for this query in the "
                "selected file(s). Tell the user you couldn't find this in "
                "the document(s) and ask if they'd like a general answer instead."
                )
            else:
                context = "\n\n".join(
                    f"[source: {meta.get('source')}, page {meta.get('page')}]\n{content}"
                    for content, meta in results
                )
                pdf_source_map = {
                    i + 1: {"source": meta.get("source"), "page": meta.get("page")}
                    for i, (content, meta) in enumerate(results)
                }


                system_prompt = f"""You are a PDF assistant. Use ONLY the numbered context blocks below to answer.

Citation rules:
- After every claim drawn from the context, add the matching marker, e.g. [1] or [2].
- Only use marker numbers that appear in the context below. Never invent a number.
- If a sentence combines multiple sources, cite all of them, e.g. [1][2].
- If the answer isn't in the context, say so first, then clearly label anything else as outside the PDF and do not cite it.

Context:
{context}"""

        elif decision.mode == "summary":
            search_query = decision.retrieve_query or query
            docs = self.summary_store.as_retriever(
                search_kwargs={"k": 2, "filter": source_filter} if source_filter else {"k": 2}
                ).invoke(search_query)
            if not docs:
                system_prompt = (
                "No document summary was found for the selected file(s). "
                "Tell the user no summary is available yet."
                )
            else:
                context = "\n\n".join(
                    f"[source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs
                )
                cited_sources = [{"source": d.metadata.get("source")} for d in docs]
                system_prompt = f"""You are answering questions about uploaded document(s) using their summaries.
                If multiple documents are relevant, compare/contrast them clearly by source name.

                Summary Context:
                {context}"""

        else:
            system_prompt = "You are a general assistant. Answer from your own knowledge or the chat history."

        messages = [SystemMessage(content=system_prompt), *self.chat_history]
        response = model.invoke(messages)
        self.chat_history.append(AIMessage(content=response.content))
        self.chat_history = _trim_history(self.chat_history)

        if pdf_source_map is not None:
            cited_sources = _extract_used_citations(response.content , pdf_source_map)

        return {
            "answer": response.content,
            "mode": decision.mode,
            "sources": cited_sources,
        }





