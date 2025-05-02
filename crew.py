import streamlit as st
from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from tasks import research_task, write_task

# Streamlit heading
st.title("AI Powered Road Side Assistance")

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[blog_researcher, blog_writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True,
   embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    }
)

# Start the task execution process with enhanced feedback
st.write("The Agents are ready to assist you with your query!")
# for _ in range(5):
topic = st.text_input("Enter the topic you want assistance with:", "Engine oil necessity in BMW cars")
result = crew.kickoff(inputs={'topic': topic})
st.write(result)