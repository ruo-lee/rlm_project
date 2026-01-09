import os
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("DEBUG: Starting main.py imports...", flush=True)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.rlm import RLMAgent
from termcolor import colored

print("DEBUG: Imports complete.", flush=True)

# Load env variables
load_dotenv(dotenv_path=".env.local")

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(colored("Error: GEMINI_API_KEY not found in .env.local", "red"))
        print("Please create .env.local and add your GEMINI_API_KEY.")
        sys.exit(1)

    print(colored("Recursive Language Model (RLM) Runner", "green", attrs=["bold"]))
    
    # 1. Prepare Context Data (NSMC - Naver Sentiment Movie Corpus)
    data_file = "ratings_train.txt"
    data_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    
    if not os.path.exists(data_file):
        print(colored(f"Downloading sample long text (NSMC) from {data_url}...", "yellow"))
        print(colored("This may take a while (14MB)...", "yellow"))
        import urllib.request
        
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = "\r%5.1f%% %*d / %d" % (
                    percent, len(str(totalsize)), readsofar, totalsize)
                sys.stdout.write(s)
                if readsofar >= totalsize: # near the end
                    sys.stdout.write("\n")
        
        try:
            urllib.request.urlretrieve(data_url, data_file, reporthook)
            print(colored("Download complete.", "green"))
        except Exception as e:
            print(colored(f"Failed to download: {e}", "red"))
            sys.exit(1)
            
    print(colored(f"Loading context from {data_file}...", "cyan"))
    with open(data_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    # For demonstration, use a subset or full text. 
    # 14MB is huge for a quick demo, let's take the first 100,000 characters (approx 20,000 lines).
    # You can increase this limit to test "Infinite Context".
    context_limit = 100000 
    sample_context = full_text[:context_limit]
    print(f"Context loaded: {len(sample_context)} characters (subset of {len(full_text)}).")
    
    # Sample Query for this dataset
    query = "이 데이터셋에서 가장 많이 등장하는 긍정적인 단어 3개를 찾아줘. 그리고 2023년이라는 숫자가 포함된 리뷰가 있는지 확인해줘."

    agent = RLMAgent()
    final_answer = agent.run(sample_context, query)
    
    print(colored(f"\nFinal Answer from RLM: {final_answer}", "green", attrs=["bold"]))

if __name__ == "__main__":
    main()
