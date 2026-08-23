# Self-RAG: Reliable Retrieval Augmented Generation

This project is a Self-RAG based question-answering system designed to make
RAG applications more reliable.

A basic RAG system usually follows:

Question → Retrieve Documents → Generate Answer

But retrieval can sometimes return irrelevant information, the required
information may be missing from the documents, or the LLM may generate an
answer that is not actually supported by the retrieved context.

This project tries to handle these problems by adding decision-making,
retrieval validation, query rewriting, web fallback, and hallucination
checking to the RAG pipeline.

## What this project does

The system takes a user's question and decides how it should be answered.

It can:

- Search internal documents for relevant information.
- Use both keyword and semantic search.
- Check whether the retrieved documents are relevant.
- Rewrite the question when retrieval is not good enough.
- Retrieve information from the web when internal documents are insufficient.
- Generate an answer using the retrieved information.
- Check whether the generated answer is actually supported by the context.
- Identify partially supported or unsupported answers.
- Retry retrieval when more information is required.
- Stop after a maximum number of retries and return a safe response instead of making up an answer.
- Expose the complete pipeline through a **FastAPI API**.
- Handle failures from LLM and external API calls.

The main idea is that the LLM should not simply answer every question from
its own knowledge. It should first use the available evidence and verify
the generated answer against that evidence.

## How the pipeline works

The overall flow is:

User Question
      ↓
Retrieval Decision
      ↓
Hybrid Retrieval
(BM25 + FAISS)
      ↓
Relevance Check
      ↓
Query Rewrite if required
      ↓
Web Retrieval if required
      ↓
Answer Generation
      ↓
Hallucination / Support Check
      ↓
Supported?
  |--> Yes → Final Answer
  |--> No
       ↓
   Rewrite Query
       ↓
   Retrieve Again
       ↓
   Generate Again
       ↓
   Check Again
       ↓
   Repeat / Stop after max retries


## Hybrid Retrieval

The project uses two types of retrieval:

### BM25

BM25 performs keyword-based retrieval. It is useful when the question contains
specific words, names, numbers, or terms that also appear in the documents.

### FAISS

FAISS is used for vector similarity search. It helps retrieve documents based
on semantic meaning rather than only exact keyword matches.

Using both approaches makes retrieval more robust than relying on only
keyword search or only vector search.


## Query Rewriting

Sometimes the original question is not ideal for document retrieval.

The system can rewrite the question into a shorter retrieval query while
preserving the important entities and intent.

For example:
"What benefits are included in the employee plan?"

can become:

"employee plan benefits coverage included"

This helps improve the chances of retrieving the correct documents.


## Self-RAG / Hallucination Checking

After generating an answer, the system checks whether it is supported by the
retrieved context.

- **Supported** → The answer is accepted.
- **Not supported** → The question is rewritten and the system retrieves
  information again.

This process is repeated for a limited number of **retries**. If a supported
answer cannot be found within the maximum retries, the system returns a safe
response instead of generating an unsupported answer.

This feedback loop helps reduce hallucinations, unsupported claims and makes the RAG system more
reliable.

## Web Retrieval

Internal documents may not always contain the information required to answer
a question.

In such cases, the system can use web retrieval as a fallback.

This allows the system to handle questions where the required information is
not available in the internal knowledge base.

The retrieved web information is also passed through the answer validation
process rather than blindly trusting the web result.

## FastAPI

The complete RAG pipeline is exposed through a FastAPI API.

Main endpoint:

    POST /query

Example request:

```json
{
  "question": "What information is available about the company?"
}