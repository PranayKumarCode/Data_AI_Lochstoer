import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_direct_dataset_search():
    """Test the MCP server with direct dataset search"""
    
    server_params = StdioServerParameters(
        command="python3.12",
        args=["mcp_server_v2.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("=" * 70)
            print("LECTURE + DATASET SEARCH ASSISTANT v4.0")
            print("Direct dataset search - No pre-training needed!")
            print("=" * 70)
            print()
            
            # Show available sources
            print("=" * 70)
            print("AVAILABLE SOURCES")
            print("=" * 70)
            
            sources_result = await session.call_tool("list_available_sources", arguments={})
            sources_data = json.loads(sources_result.content[0].text)
            
            print(f"\n📚 Lectures: {sources_data['lectures']['count']} files, {sources_data['lectures']['total_chunks']} chunks")
            for f in sources_data['lectures']['files'][:5]:
                print(f"   • {f}")
            if len(sources_data['lectures']['files']) > 5:
                print(f"   ... and {len(sources_data['lectures']['files']) - 5} more")
            
            print(f"\n📊 Datasets: {len(sources_data['datasets'])} datasets")
            for ds in sources_data['datasets']:
                print(f"   • {ds['name']}: {ds['examples']:,} examples")
            
            print(f"\n✓ Total searchable examples: {sources_data['total_examples']:,}")
            
            # Test questions
            test_queries = [
                "What is a decision tree?",
                "What is the Black-Scholes equation?",
                "What is gradient descent?",
                "What is unsupervised learning?",
                "How does cross-validation work?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n\n{'='*70}")
                print(f"TEST {i}: {query}")
                print('='*70)
                
                # Use ask_question tool
                result = await session.call_tool(
                    "ask_question",
                    arguments={"question": query}
                )
                
                data = json.loads(result.content[0].text)
                
                if data.get("answer") and data["answer"] != "Not found in provided materials":
                    print(f"\n✓ ANSWER: {data['answer']}")
                    print(f"\n  Confidence: {data.get('confidence', 0.0):.2f} / 1.00")
                    print(f"  Query Type: {data.get('query_type', 'unknown').upper()}")
                    
                    print(f"\n  Sources Used:")
                    sources_breakdown = data.get('sources_breakdown', {})
                    print(f"    • Lectures: {sources_breakdown.get('lectures', 0)} results")
                    print(f"    • Datasets: {sources_breakdown.get('datasets', 0)} results")
                    
                    if data.get('sources'):
                        print(f"\n  Top Sources:")
                        for source in data['sources'][:3]:
                            print(f"    {source}")
                else:
                    print(f"\n✗ {data.get('message', 'No answer found')}")
                    if data.get('searched'):
                        print(f"\n  Searched:")
                        searched = data['searched']
                        print(f"    • {searched.get('lectures', 0)} lecture chunks")
                        print(f"    • {searched.get('datasets', 0)} datasets")
                        print(f"    • {searched.get('total_dataset_examples', 0):,} total examples")
                
                await asyncio.sleep(0.5)
            
            # Test detailed search
            print(f"\n\n{'='*70}")
            print("DETAILED SEARCH EXAMPLE")
            print('='*70)
            
            detailed_query = "What is the difference between supervised and unsupervised learning?"
            print(f"\nQuery: {detailed_query}\n")
            
            search_result = await session.call_tool(
                "search_all_sources",
                arguments={
                    "query": detailed_query,
                    "k_lectures": 3,
                    "k_datasets": 5
                }
            )
            
            search_data = json.loads(search_result.content[0].text)
            
            print("="*70)
            print("EXTRACTED ANSWER")
            print("="*70)
            extracted = search_data.get('extracted_answer', {})
            print(f"\n{extracted.get('answer', 'N/A')}")
            print(f"\nConfidence: {extracted.get('confidence', 0.0):.2f}")
            print(f"Sources: {extracted.get('sources', {})}")
            
            print("\n" + "="*70)
            print("LECTURE RESULTS")
            print("="*70)
            for i, r in enumerate(search_data['lecture_results'], 1):
                print(f"\n[{i}] {r['file']} (Page {r['page']})")
                print(f"    Distance: {r['distance']:.4f}")
                print(f"    {r['content'][:200]}...")
            
            print("\n" + "="*70)
            print("DATASET RESULTS")
            print("="*70)
            for i, r in enumerate(search_data['dataset_results'], 1):
                print(f"\n[{i}] {r['dataset_name']}")
                print(f"    Score: {r['score']}")
                print(f"    Q: {r['question'][:100]}...")
                print(f"    A: {r['answer'][:200]}...")
            
            print("\n" + "="*70)
            print("TESTING COMPLETE")
            print("="*70)
            print("\n✓ System can now search across:")
            print(f"  • {sources_data['lectures']['total_chunks']} lecture chunks")
            print(f"  • {sources_data['total_examples'] - sources_data['lectures']['total_chunks']:,} dataset examples")
            print(f"  • Total: {sources_data['total_examples']:,} searchable items")
            print("\n✓ No pre-training required!")
            print("✓ Instant startup!")
            print("✓ All examples accessible!")

if __name__ == "__main__":
    asyncio.run(test_direct_dataset_search())