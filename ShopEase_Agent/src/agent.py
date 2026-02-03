"""
ShopEase Agentic Analyst (Hybrid Mode)
Supports both LangChain (OpenAI) and Mock (Rule-Based) modes for demo purposes.
"""
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from .tools import (
    get_data_summary, 
    analyze_delivery_impact, 
    analyze_channel_performance,
    analyze_city_performance,
    analyze_demographics,
    analyze_engagement,
    train_predictive_model
)
from .rag import get_business_context

# Load env variables (API Keys)
load_dotenv()

class ShopEaseAnalyst:
    def __init__(self):
        # 1. Initialize Tools
        self.tools = [
            get_business_context,
            get_data_summary, 
            analyze_delivery_impact, 
            analyze_channel_performance,
            analyze_city_performance,
            analyze_demographics,
            analyze_engagement,
            train_predictive_model
        ]
        
        # Check API Key to decide mode
        api_key = os.getenv("OPENAI_API_KEY")
        self.use_mock = False
        
        if not api_key or api_key.startswith("paste_your"):
            print("[SYSTEM] No valid OPENAI_API_KEY found. Switching to MOCK AGENT mode.")
            self.use_mock = True
        else:
            try:
                # Local import to prevent crash if library is missing/incompatible
                from langchain.agents import create_openai_tools_agent, AgentExecutor
                
                # 2. System Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an expert Data Analyst Agent for ShopEase. "
                     "Your goal is to answer business questions by querying the Knowledge Base for context "
                     "and using the statistical tools for quantitative analysis. "
                     "Always start by checking the business context if the question implies a 'why' or 'problem'. "
                     "When a plot is generated, mention the visualization path in your final answer."),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                # 3. Create Agent
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
                agent = create_openai_tools_agent(llm, self.tools, prompt)
                self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            except ImportError:
                print("[SYSTEM] LangChain version incompatible. Switching to MOCK AGENT mode.")
                self.use_mock = True
            except Exception as e:
                print(f"[SYSTEM] Error initializing OpenAI Agent: {e}. Switching to MOCK AGENT mode.")
                self.use_mock = True

    def process_query(self, query: str):
        """
        Main entry point. Routes to either LangChain or Mock implementation.
        """
        print(f"\n[AGENT] Received Query: '{query}'")
        
        if self.use_mock:
            return self._mock_process_query(query)
        else:
            try:
                response = self.agent_executor.invoke({"input": query})
                return response["output"]
            except Exception as e:
                print(f"[ERROR] LangChain execution failed: {e}. Falling back to Mock.")
                return self._mock_process_query(query)

    def _mock_process_query(self, query: str):
        """
        Simulates the Agent's reasoning for demo purposes without an LLM.
        """
        print("[MOCK-AGENT] Consulting Knowledge Base...")
        context = get_business_context.invoke(query)
        print(f"[MOCK-AGENT] Retrieved Context: {context[:50]}...")
        
        q = query.lower()
        result = ""
        plan = "check context"
        
        # Simple Keyword Routing (The "Planner")
        if "delivery" in q:
            print(f"[MOCK-AGENT] Decided to call tool: analyze_delivery_impact")
            result = analyze_delivery_impact.invoke({})
            plan = "investigate delivery times"
        elif "channel" in q:
            print(f"[MOCK-AGENT] Decided to call tool: analyze_channel_performance")
            result = analyze_channel_performance.invoke({})
            plan = "check acquisition channels"
        elif "city" in q or "region" in q:
            print(f"[MOCK-AGENT] Decided to call tool: analyze_city_performance")
            result = analyze_city_performance.invoke({})
            plan = "check city breaks"
        elif "gender" in q or "demograph" in q:
            print(f"[MOCK-AGENT] Decided to call tool: analyze_demographics")
            result = analyze_demographics.invoke({})
        elif "visit" in q or "engage" in q:
            print(f"[MOCK-AGENT] Decided to call tool: analyze_engagement")
            result = analyze_engagement.invoke({})
        elif "model" in q or "predict" in q or "driver" in q:
            print(f"[MOCK-AGENT] Decided to call tool: train_predictive_model")
            result = train_predictive_model.invoke({})
            plan = "train ML model"
        else:
            print(f"[MOCK-AGENT] Decided to call tool: get_data_summary")
            result = get_data_summary.invoke({})
            
        final_response = f"""
--- AGENT RESPONSE (MOCK MODE) ---
Based on the Mock Logic: The user asked about '{plan}'.

Analysis Result:
{result}

Recommendation:
(Simulated) Please investigate the highlighted metrics above, specifically targeting high-churn segments.
----------------------
"""
        return final_response

if __name__ == "__main__":
    agent = ShopEaseAnalyst()
    print(agent.process_query("Does delivery time affect churn?"))
