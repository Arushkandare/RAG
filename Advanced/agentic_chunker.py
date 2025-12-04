from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI

class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True

        self.llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)
    
    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({chunk_id}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("No chunks found")
            self._create_new_chunk(proposition)
    
    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """You are grouping similar sentences. Generate a 1-sentence summary that generalizes the topic of all given propositions. Be concise and clear."""),
            ("user", "Propositions:\n{proposition}\n\nCurrent summary:\n{current_summary}")
        ])
        runnable = PROMPT | self.llm

        result = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        }).content.strip()

        return result

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """You are generating titles for groups of related sentences. Given the summary and propositions, generate a concise and generalized title (e.g., 'Food Preferences', 'Dates & Time')."""),
            ("user", "Propositions:\n{proposition}\n\nSummary:\n{current_summary}\n\nCurrent title:\n{current_title}")
        ])
        runnable = PROMPT | self.llm

        result = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content.strip()

        return result

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Summarize the following proposition into a 1-sentence general summary suitable for grouping similar statements."""),
            ("user", "Proposition:\n{proposition}")
        ])
        runnable = PROMPT | self.llm
        result = runnable.invoke({"proposition": proposition}).content.strip()
        return result

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Generate a short and generalized title based on the summary below (e.g., 'Dates & Time', 'Food Preferences')."""),
            ("user", "Summary:\n{summary}")
        ])
        runnable = PROMPT | self.llm
        result = runnable.invoke({"summary": summary}).content.strip()
        return result

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }

        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing whether a new proposition belongs to an existing chunk based on semantic similarity and topical relevance.

Your task:
1. Compare the new proposition with each existing chunk's title and summary
2. Determine if the proposition is semantically related or topically similar to any chunk
3. Return ONLY the chunk ID if there's a good match, or return "NONE" if no match exists

Guidelines:
- Look for thematic connections, not just exact keyword matches
- Consider if the proposition would logically belong with the existing content
- If unsure, err on the side of creating a new chunk (return "NONE")

Response format: Return only the chunk ID (e.g., "d41a2") or "NONE"""
            ),
            ("user", "Existing Chunks:\n{current_chunk_outline}\n\nNew Proposition: {proposition}")
        ])

        runnable = PROMPT | self.llm
        result = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content.strip()

        if self.print_logging:
            print(f"LLM Response: '{result}'")

        # More robust checking for "no match" responses
        if result.upper() in ["NONE", "NO CHUNKS", "NO MATCH", "NO"]:
            return None
        
        # Check if the result is a valid chunk ID
        if result in self.chunks:
            return result
        
        # Check if it's a truncated version of a valid chunk ID
        if result[:self.id_truncate_limit] in self.chunks:
            return result[:self.id_truncate_limit]
        
        # Try to find a chunk ID within the response (in case LLM adds extra text)
        for chunk_id in self.chunks.keys():
            if chunk_id in result:
                return chunk_id
        
        return None

    def get_chunk_outline(self):
        outline = ""
        for chunk in self.chunks.values():
            outline += f"Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"
        return outline

    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]

    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk in self.chunks.values():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Title: {chunk['title']}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in chunk['propositions']:
                print(f"  - {prop}")
            print("\n")

    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())


if __name__ == "__main__":
    ac = AgenticChunker()

    propositions = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
    ]
    
    ac.add_propositions(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks(get_type='list_of_strings'))