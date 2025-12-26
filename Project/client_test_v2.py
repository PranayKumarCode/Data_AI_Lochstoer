import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_and_interactive():
    """Run tests then go interactive"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_v2.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("=" * 70)
            print("AI-ENHANCED LECTURE ASSISTANT")
            print("Ollama-powered with query expansion, reranking & synthesis")
            print("=" * 70)
            print()
            
            # Show sources
            print("=" * 70)
            print("LOADING SOURCES")
            print("=" * 70)
            
            sources_result = await session.call_tool("list_available_sources", arguments={})
            sources_data = json.loads(sources_result.content[0].text)
            
            print(f"\n📚 Lectures: {sources_data['lectures']['count']} files, {sources_data['lectures']['total_chunks']} chunks")
            print(f"📊 Datasets: {len(sources_data['datasets'])} datasets")
            for ds in sources_data['datasets']:
                print(f"   • {ds['name']}: {ds['examples']:,} examples")
            print(f"\n✓ Total: {sources_data['total_examples']:,} searchable items")
            
            # Run automatic tests
            print("\n" + "=" * 70)
            print("RUNNING AUTOMATIC TESTS")
            print("=" * 70)
            
            test_queries = [
                "What is a decision tree?",
                "What is gradient descent?",
                "What is the difference between supervised and unsupervised learning?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n{'='*70}")
                print(f"TEST {i}: {query}")
                print('='*70)
                
                result = await session.call_tool(
                    "search_all_sources",
                    arguments={
                        "query": query,
                        "k_lectures": 20,
                        "k_datasets": 20,
                        "use_ai_enhancement": True
                    }
                )
                
                data = json.loads(result.content[0].text)
                
                if data.get("answer") and "ERROR:" not in data["answer"]:
                    print(f"\n✓ ANSWER:")
                    print(f"  {data['answer']}")
                    print(f"\n  Confidence: {data.get('confidence', 0.0):.2f}")
                    print(f"  Method: {data.get('extraction_method', 'unknown')}")
                    print(f"  AI Enhanced: {data.get('ai_enhanced', False)}")
                    
                    if data.get('sources'):
                        print(f"\n  Sources:")
                        for source in data['sources'][:3]:
                            print(f"    {source}")
                else:
                    print(f"\n✗ {data.get('answer', 'No answer')}")
                
                await asyncio.sleep(0.5)
            
            # Interactive mode
            print("\n\n" + "=" * 70)
            print("INTERACTIVE MODE")
            print("=" * 70)
            print("\nAutomatic tests complete! Now you can ask your own questions.")
            print("\nCommands:")
            print("  Type your question - Get an AI-synthesized answer")
            print("  'detailed <question>' - See full search results")
            print("  'quit' or 'exit' - Exit program")
            print("=" * 70)
            
            while True:
                try:
                    question = input("\n❓ Your question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Detailed search
                if question.lower().startswith('detailed '):
                    question = question[9:].strip()
                    print(f"\n🔍 Detailed search: {question}")
                    print("=" * 70)
                    
                    result = await session.call_tool(
                        "search_all_sources",
                        arguments={
                            "query": question,
                            "k_lectures": 20,
                            "k_datasets": 20,
                            "use_ai_enhancement": True
                        }
                    )
                    
                    data = json.loads(result.content[0].text)
                    
                    # Show AI enhancements
                    enhancements = data.get('ai_enhancements', {})
                    if enhancements.get('expanded_terms'):
                        print(f"\n🤖 AI expanded query to: {', '.join(enhancements['expanded_terms'])}")
                    
                    # Show answer
                    extracted = data.get('extracted_answer', {})
                    print(f"\n✓ ANSWER:")
                    print(f"  {extracted.get('answer', 'Not found')}")
                    print(f"\n  Confidence: {extracted.get('confidence', 0.0):.2f}")
                    print(f"  Method: {extracted.get('type', 'unknown')}")
                    
                    # Show lecture results
                    if data.get('lecture_results'):
                        print(f"\n📄 LECTURE RESULTS:")
                        for i, r in enumerate(data['lecture_results'][:3], 1):
                            print(f"\n  [{i}] {r['file']} (Page {r['page']})")
                            print(f"      Relevance: {1 - r['distance']:.2f}")
                            print(f"      {r['content'][:200]}...")
                    
                    # Show dataset results
                    if data.get('dataset_results'):
                        print(f"\n📊 DATASET RESULTS:")
                        for i, r in enumerate(data['dataset_results'][:3], 1):
                            print(f"\n  [{i}] {r['dataset_name']} (Score: {r['score']})")
                            print(f"      Q: {r['question'][:100]}...")
                            print(f"      A: {r['answer'][:150]}...")
                    
                    continue
                
                # Normal question
                print(f"\n🔍 Processing: {question}")
                
                try:
                    result = await session.call_tool(
                        "ask_question",
                        arguments={"question": question}
                    )
                    
                    data = json.loads(result.content[0].text)
                    
                    if "ERROR:" in data.get("answer", ""):
                        print(f"\n❌ {data['answer']}")
                        print("\n💡 Make sure Ollama is running:")
                        print("   ollama serve")
                    elif data.get("answer") and data["answer"] != "Not found in provided materials":
                        print(f"\n✓ ANSWER:")
                        print(f"  {data['answer']}")
                        print(f"\n  Confidence: {data.get('confidence', 0.0):.2f}")
                        print(f"  Query Type: {data.get('query_type', 'unknown').upper()}")
                        print(f"  AI Enhanced: ✓")
                        
                        if data.get('sources'):
                            print(f"\n  📚 Sources:")
                            for source in data['sources'][:4]:
                                print(f"    {source}")
                        
                        print(f"\n💡 Try 'detailed {question}' for full search results")
                    else:
                        print(f"\n✗ Could not find answer in provided materials")
                        print(f"\n💡 Try rephrasing or use 'detailed {question}'")
                
                except Exception as e:
                    print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("\nStarting AI-Enhanced Assistant...")
    print("(Loading datasets and models... ~10 seconds)\n")
    asyncio.run(test_and_interactive())