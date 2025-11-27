import asyncio
import os
import json
from src.storage import MinioHandler
from src.workflow import build_graph
from src.config import logger
from src.agent_hiring import HiringAgent

async def main():
    minio = MinioHandler()
    await minio.ensure_bucket()

    logger.info("--- 🚀 Iran HR Agent System ---")
    
    # 1. User Input Phase for Upload
    choice = input("Do you want to upload resumes from a local folder? (y/n): ").strip().lower()
    
    if choice == 'y':
        folder_path = input("Enter the absolute path to your resume folder: ").strip()
        if os.path.exists(folder_path):
            logger.info("Uploading files... this might take a moment.")
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            
            # Upload concurrently to speed up initialization
            upload_tasks = []
            for f in files:
                full_path = os.path.join(folder_path, f)
                upload_tasks.append(minio.upload_file(full_path, f))
            
            if upload_tasks:
                await asyncio.gather(*upload_tasks)
                logger.info(f"Done. Uploaded {len(files)} files.")
            else:
                logger.info("No PDF files found in folder.")
        else:
            logger.info("❌ Folder not found. Skipping upload.")
    
    logger.info("\n--- 🗣️ Step 1: Hiring Requirement Interview ---")
    
    hiring_agent = HiringAgent()
    
    # Initial Greeting
    print("\n🤖 Agent: Hello! I am your AI Recruiter. What kind of position are you hiring for today?")
    
    while not hiring_agent.is_complete:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            logger.info("User aborted conversation.")
            return

        response_text = await hiring_agent.run_turn(user_input)
        print(f"🤖 Agent: {response_text}")
    
    # Result of Phase 2
    reqs = hiring_agent.final_requirements
    logger.info(f"\n✅ Requirements Captured: {reqs.role_title} ({reqs.seniority})")
    logger.info(f"   Skills: {reqs.essential_hard_skills}")
    logger.info(f"   Military Service Required: {reqs.military_service_required}")
    
    # Save requirements to file (so the workflow could theoretically use them later)
    with open("hiring_requirements.json", "w", encoding="utf-8") as f:
        f.write(reqs.model_dump_json(indent=2))

    # 2. Workflow Execution Phase
    logger.info("\n--- ⚡ Step 2: Starting Resume Processing Workflow ---")
    app = build_graph()
    
    # Initial state
    inputs = {"file_keys": [], "results": [], "errors": []}
    
    # Run the graph
    final_state = await app.ainvoke(inputs)
    
    # 3. Summary & Save
    errors = final_state.get("errors", []) 
    
    # Save to JSON file
    output_file = "parsed_resumes.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2)
        
    print("\n--- 🏁 Execution Finished ---")
    print(f"Total Processed: {len(final_state.get('final_results',[]))}")
    print(f"Total Errors: {len(errors)}")
    print(f"Results saved to {output_file}")
    
    if errors:
        print("Check 'errors' list in logs for failed files.")

if __name__ == "__main__":
    asyncio.run(main())