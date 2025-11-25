import asyncio
import os
import json
from src.storage import MinioHandler
from src.workflow import build_graph

async def main():
    minio = MinioHandler()
    await minio.ensure_bucket()

    print("--- 🚀 Iran HR Resume Parser Agent ---")
    
    # 1. User Input Phase for Upload
    choice = input("Do you want to upload resumes from a local folder? (y/n): ").strip().lower()
    
    if choice == 'y':
        folder_path = input("Enter the absolute path to your resume folder: ").strip()
        if os.path.exists(folder_path):
            print("Uploading files... this might take a moment.")
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            
            # Upload concurrently to speed up initialization
            upload_tasks = []
            for f in files:
                full_path = os.path.join(folder_path, f)
                upload_tasks.append(minio.upload_file(full_path, f))
            
            if upload_tasks:
                await asyncio.gather(*upload_tasks)
                print(f"Done. Uploaded {len(files)} files.")
            else:
                print("No PDF files found in folder.")
        else:
            print("❌ Folder not found. Skipping upload.")
    
    # 2. Workflow Execution Phase
    print("\n--- ⚡ Starting Workflow ---")
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