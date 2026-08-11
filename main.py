from dotenv import load_dotenv
from utils.Audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize,generate_title
from core.extractor import extract_action_items,extract_key_decisions,extract_questions
from core.rag_engine import build_rag_chain,ask_question
load_dotenv()

def run_pipeline(source:str , language:str="english")-> dict:
    print("Starting AI Video Assistant")
    
    chunks=process_input(source)
    
    transcript=transcribe_all(chunks,language=language)
    print(f"raw transcript (first 300 character){transcript[:300]}")
    
    title = generate_title(transcript)
    
    summary = summarize(transcript)
    
    action_item = extract_action_items(transcript)
    
    desicisions = extract_key_decisions(transcript)
    
    questions= extract_questions(transcript)
    
    rag_chain= build_rag_chain(transcript)
    
    return {
        "title": title,
        "summary": summary,
        "transcript": transcript,
        "action_item": action_item,
        "key_decision": desicisions,
        "open_question": questions,
        "rag_chain": rag_chain,
    }
    
if __name__ == "__main__":
    
    source = input("Enter Youtube URL or Local file path->").strip()
    language = input("Language (english/hinglish):").strip() or "english"
    result = run_pipeline(source,language)
    
    print("/n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_item']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decision']}")
    print(f"\n❓ Open Questions:\n{result['open_question']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")