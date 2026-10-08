"""Bước 2: push/pull Prompt Hub, MD5 routing và 50 A/B traces."""
import config
import hashlib
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError
from prompts import (
    PROMPT_V1_NAME, PROMPT_V2_NAME, SYSTEM_V1, SYSTEM_V2, PROMPT_V1, PROMPT_V2,
)
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.evidence import capture_log
from qa_pairs import SAMPLE_QUESTIONS


def push_prompts_to_hub(client: Client):
    """Push hai prompt riêng tư; chỉ bỏ qua conflict Nothing to commit."""
    for name, prompt, description in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1: concise, context-only tutor"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2: structured, evidence-focused analyst"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"✅ Đã push {name} → {url}")
        except LangSmithConflictError as error:
            if "nothing to commit" not in str(error).lower():
                raise
            print(f"ℹ️ {name}: Nothing to commit, giữ phiên bản đã có.")


def pull_prompts_from_hub(client: Client) -> dict:
    """Pull thật từ Hub; không coi prompt local là evidence của Hub."""
    prompts = {}
    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        prompt = client.pull_prompt(name)
        if set(prompt.input_variables) != {"context", "question"}:
            raise ValueError(f"Prompt {name} phải có context và question")
        prompts[name] = prompt
        print(f"↓ Đã pull '{name}' từ Hub")
    return prompts


def get_prompt_version(request_id: str) -> str:
    """Cùng request_id luôn nhận cùng prompt, kể cả ở process khác."""
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Trả context riêng lẻ trong trace để kiểm tra căn cứ của answer."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": "\n\n".join(contexts), "question": question,
    })
    return {"question": question, "answer": answer, "version": version,
            "contexts": contexts}


def setup_vectorstore():
    """Tạo cùng cấu hình truy xuất với bước 1 và 3."""
    return build_vectorstore(split_text(load_knowledge_base()), get_embeddings())


def main():
    """Log request_id, nhãn prompt, câu hỏi và answer vào evidence thật."""
    if not config.validate():
        raise SystemExit(1)
    with capture_log("02_ab_routing_log.txt"):
        print("Bước 2: Prompt Hub & A/B Routing")
        client = Client(api_key=config.LANGSMITH_API_KEY)
        push_prompts_to_hub(client)
        prompts = pull_prompts_from_hub(client)
        retriever = setup_vectorstore().as_retriever(search_kwargs={"k": 3})
        llm = get_llm()
        counts = {"v1": 0, "v2": 0}
        try:
            for i, question in enumerate(SAMPLE_QUESTIONS):
                request_id = f"req-{i:04d}"
                version_key = get_prompt_version(request_id)
                version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
                result = ask_ab(retriever, llm, prompts[version_key], question,
                                version_tag, langsmith_extra={
                                    "client": client,
                                    "metadata": {"request_id": request_id,
                                                 "prompt_name": version_key},
                                })
                counts[version_tag] += 1
                print(f"[{i+1:02d}] {request_id} [prompt-{version_tag}] Q: {question}")
                print(f"     A: {result['answer']}")
        finally:
            client.flush()
        if not all(counts.values()):
            raise RuntimeError("Cần có câu hỏi được gửi tới cả hai phiên bản")
        print(f"📊 Routing: V1={counts['v1']} | V2={counts['v2']} | "
              f"Tổng={sum(counts.values())}")
        print("✅ Hoàn thành gọi A/B. Xác nhận Prompt Hub và traces trong LangSmith UI.")


if __name__ == "__main__":
    main()
