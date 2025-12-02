import asyncio
import os
import json
from langgraph.types import Command
from src.storage import MinioHandler
from src.workflow import build_graph
from src.config import logger
from src.agent_hiring import HiringAgent
from src.matcher import ResumeQAAgent
from src.database import MongoHandler
from src.config import config
from utils.extract_structure import ExtractSchema
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import random

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
    
    app = build_graph()
    thread_config = {"configurable": {"thread_id": "unified_session_1"}}

    # Initial state
    inputs = {
        "all_files": [], 
        "hiring_reqs": [], # Inject requirements
        "final_results": [],
        "db_structure": {},
        "hiring_messages": [HumanMessage('Introduce yourself.')]
    }
    logger.info("Starting System...")
    parser = StrOutputParser()

    async for event in app.astream(inputs, config=thread_config):
        if "hiring_process" in event:
            msgs = event["hiring_process"].get("hiring_messages", [])
            if msgs:
                last_msg = msgs[-1]
                # If it's an AI message with text (and not a tool call hidden logic), print it
                if hasattr(last_msg, 'content') and last_msg.content:
                    print(f"\n🤖 Agent: {parser.invoke(last_msg)}")
    

    # --- PHASE 6: Q&A SYSTEM ---
    # logger.info("\n--- 💬 Database Q&A (Type 'exit' to quit) ---")
    # print("Ask questions about the resumes (e.g., 'Who knows Python?', 'Is Morteza here?').")
    while True:
        # Check current state to see if we are indeed paused at 'qa_input'
        snapshot = await app.aget_state(thread_config)
        
        if not snapshot.next:
            logger.info("✅ Workflow Finished.")
            break
        
        # Check for Interrupts
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            # The 'value' we passed to interrupt() is here
            interrupt_value = snapshot.tasks[0].interrupts[0].value
            
            # Context-aware Prompting
            if interrupt_value == "hiring_input":
                user_input = input("\n👤 You (Requirement): ").strip()
            elif interrupt_value == "qa_input":
                user_input = input("\n❓ You (Question): ").strip()
            else:
                user_input = input("\n👤 Input needed: ").strip()
            
            # Resume Graph
            async for event in app.astream(Command(resume=user_input), config=thread_config):
                if "hiring_process" in event:
                    msgs = event["hiring_process"].get("hiring_messages", [])
                    if msgs:
                        last_msg = msgs[-1]
                        if hasattr(last_msg, 'content') and last_msg.content:
                            print(f"\n🤖 Agent: {parser.invoke(last_msg)}")

        else:
            # Should not happen if graph is designed correctly with interrupts
            break

if __name__ == "__main__":
    asyncio.run(main())