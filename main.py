import chainlit as cl
import asyncio
import os
from pymongo import MongoClient

# Graph & Service Imports
from langgraph.types import Command
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Your existing project imports
from app.config.logger import logger
from app.services.minio_service import MinioHandler
from app.workflow.builder import build_graph
from app.config.config import config

mongo_client = MongoClient(config.mongo_uri)


# --- HELPER: Initialize MinIO & Mongo once ---
parser = StrOutputParser()
minio = MinioHandler()
# We create the checkpointer instance. 
# Note: In production, manage the Mongo client connection lifecycle carefully.
checkpointer = MongoDBSaver(mongo_client)

@cl.on_chat_start
async def start():
    """
    Called when the user opens the page.
    1. Sets up the environment.
    2. Handles the initial file upload.
    3. Starts the graph.
    """
    await minio.ensure_bucket()

    # --- Step 1: Initial Upload (Replaces your initial input logic) ---
    res = await cl.AskActionMessage(
        #Do you want to upload resumes from a local folder now
        content="آیا می‌خواهید رزومه‌ها را از یک پوشه آپلود کنید؟",
        actions=[
            cl.Action(name="yes", payload={"value":"yes"}, label="بله, آپلود میکنم"),
            cl.Action(name="no", payload={"value":"no"}, label="نه, فعلا"),
        ]
    ).send()

    uploaded_keys = []
    
    if res and res.get("payload",{}).get("value") == "yes":
        # Pops up the File Uploader UI
        files = await cl.AskFileMessage(
            content="Please drop your PDF resumes here.",
            accept=["application/pdf"],
            max_size_mb=20,
            max_files=10
        ).send()
        
        if files:
            msg = cl.Message(content=f"Uploading {len(files)} files...")
            await msg.send()
            
            # Upload to MinIO
            tasks = [minio.upload_file(f.path, f.name) for f in files]
            await asyncio.gather(*tasks)
            uploaded_keys = [f.name for f in files]
            
            msg.content = f"✅ Done. Uploaded {len(files)} files."
            await msg.update()

    # --- Step 2: Build & Compile Graph ---
    builder = build_graph()
    # We compile the graph using the global checkpointer
    graph = builder.compile(checkpointer=checkpointer)
    
    # Store graph and config in session for reuse
    thread_id = f"session_{cl.user_session.get('id')}"
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    cl.user_session.set("graph", graph)
    cl.user_session.set("config", thread_config)

    # --- Step 3: Start the Graph ---
    initial_inputs = {
        "all_files": uploaded_keys,
        "start_message": [HumanMessage(content='Introduce yourself.')]
    }
    
    # Run the cycle
    await run_graph_cycle(initial_inputs)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Triggered when the user types text (QA input or General Response).
    """
    # Resume the graph with the text provided by the user
    await run_graph_cycle(Command(resume=message.content))


async def run_graph_cycle(input_data):
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")
    
    # We use stream_mode="updates" to see what each node just finished doing
    async for event in graph.astream(input_data, config=config, stream_mode="updates"):
        
        # event is a dict like: {'NodeName': {'key_updated': value}}
        for node_name, updates in event.items():
            
            # --- 1. Handle Chat Message Lists ---
            # We look for ANY key that holds a list of messages
            for message_key in ["start_message", "jd_messages", "hiring_messages"]:
                if message_key in updates:
                    new_msgs = updates[message_key]
                    # 'new_msgs' is usually a list. Get the last one.
                    if isinstance(new_msgs, list) and new_msgs:
                        last_msg = new_msgs[-1]
                        # Only print AI messages to the UI (skip User messages to avoid duplicates)
                        if last_msg.type == "ai" and last_msg.content:
                            await cl.Message(content=parser.invoke(last_msg)).send()

            # --- 2. Handle String Outputs (Final Results) ---
            # Sometimes your nodes return a raw string instead of a message object
            if "final_jd" in updates:
                # Use a Step for large generated text or a Message
                await cl.Message(content=f"**Generated Job Description:**\n\n{updates['final_jd']}").send()
            
            if "qa_answer" in updates:
                await cl.Message(content=updates["qa_answer"]).send()
                
            if "compare_qa_answer" in updates:
                await cl.Message(content=updates["compare_qa_answer"]).send()

            # --- 3. Debug/Feedback (Optional) ---
            # If a file upload node finished, you might see 'all_files' update
            if "all_files" in updates and updates["all_files"]:
                # Just a small toast notification
                await cl.Message(content=f"waiting for 📂 Processed {len(updates['all_files'])} files...").send()

    # 2. Check for Interrupts (The Logic Replacement for 'while True')
    snapshot = await graph.aget_state(config)
    
    if snapshot.next and snapshot.tasks[0].interrupts:
        interrupt_val = snapshot.tasks[0].interrupts[0].value
        
        # --- SCENARIO A: Graph wants a FILE UPLOAD ---
        if isinstance(interrupt_val, dict) and interrupt_val.get("type") == "compare_upload":
            
            # Show the bot's request message
            bot_request = interrupt_val.get('msg', 'Please upload a file.')
            await cl.Message(content=f"🤖 {bot_request}").send()
            
            # TRIGGER THE UI FILE UPLOADER
            files = await cl.AskFileMessage(
                content="Upload the PDF file(s) for comparison.",
                accept=["application/pdf"],
                max_size_mb=20,
                max_files=5,
                timeout=600 # Wait 10 mins for user
            ).send()
            
            if files:
                # Upload logic
                temp_msg = cl.Message(content="Uploading to MinIO...")
                await temp_msg.send()
                
                tasks = [minio.upload_file(f.path, f.name) for f in files]
                await asyncio.gather(*tasks)
                uploaded_keys = [f.name for f in files]
                
                temp_msg.content = f"Uploaded {uploaded_keys}. Resuming analysis..."
                await temp_msg.update()
                
                # RECURSIVE CALL: Resume immediately with the file keys!
                # We do NOT wait for the user to type anything.
                await run_graph_cycle(Command(resume=uploaded_keys))
            else:
                await cl.Message(content="❌ Upload timed out or cancelled.").send()

        # --- SCENARIO B: Graph wants TEXT INPUT ---
        elif "qa_input" in str(interrupt_val) or True:
            # If it's just a text question, we do NOTHING here.
            # We simply let the function finish. 
            # The UI waits for the user to type in the chat bar.
            # When they type, @cl.on_message triggers.
            
            # Optional: Print the prompt if provided
            if isinstance(interrupt_val, dict) and "msg" in interrupt_val:
                 await cl.Message(content=f"❓ {interrupt_val['msg']}").send()
            pass





