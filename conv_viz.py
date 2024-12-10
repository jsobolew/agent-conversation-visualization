import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
import re

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def normalize_text(text):
    """Normalize text formatting by removing extra asterisks and consistent casing"""
    if not text:
        return text
    # Remove markdown bold formatting (double asterisks)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return text

def display_tool_content(content):
    """Display tool content without markdown interpretation"""
    if content:
        # Use st.text() instead of st.write() to avoid markdown interpretation
        st.text(content)

def parse_conversation(data):
    """Parse conversation data"""
    try:
        messages = data if isinstance(data, list) else []
        if not messages:
            st.error("Invalid conversation format. Expected a list of messages.")
            return []
        
        # Validate message format
        for msg in messages:
            if 'content' not in msg or 'role' not in msg:
                st.error("Invalid message format. Each message must have 'content' and 'role' fields.")
                return []
                
        return messages
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return []

def analyze_conversation(messages):
    """Analyze conversation metrics"""
    if not messages:
        return None, None, None, None
    
    # Convert messages to DataFrame
    df = pd.DataFrame(messages)
    
    # Calculate metrics
    message_counts = df['role'].value_counts()
    
    # Count tool calls
    tool_calls_count = sum(1 for msg in messages if 'tool_calls' in msg and msg['tool_calls'])
    tool_responses_count = sum(1 for msg in messages if 'tool_responses' in msg and msg['tool_responses'])
    
    # Calculate average message length by role
    avg_length = df.groupby('role')['content'].apply(lambda x: x.str.len().mean())
    
    # Get tool call distribution
    tool_calls = []
    for msg in messages:
        if 'tool_calls' in msg and msg['tool_calls']:
            for tool_call in msg['tool_calls']:
                if 'function' in tool_call:
                    tool_calls.append(tool_call['function'].get('name', 'unknown'))
    
    tool_distribution = pd.Series(tool_calls).value_counts() if tool_calls else None
    
    return message_counts, avg_length, tool_calls_count, tool_distribution

def main():
    st.title("LLM Conversation Visualization Dashboard")
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("Upload Conversation")
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                messages = parse_conversation(data)
                if messages:
                    st.session_state.messages = messages
                    st.session_state.conversation_history = messages
                    st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Main content area
    if st.session_state.messages:
        # Analysis section
        st.header("Conversation Analysis")
        message_counts, avg_length, tool_calls_count, tool_distribution = analyze_conversation(st.session_state.messages)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.messages))
        with col2:
            st.metric("Tool Calls", tool_calls_count)
        with col3:
            st.metric("Average Message Length", 
                     f"{int(pd.DataFrame(st.session_state.messages)['content'].str.len().mean())}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Message Distribution")
            if message_counts is not None:
                fig = px.pie(values=message_counts.values, 
                           names=message_counts.index,
                           title="Messages by Role")
                st.plotly_chart(fig)
        
        with col2:
            st.subheader("Tool Usage Distribution")
            if tool_distribution is not None and not tool_distribution.empty:
                fig = px.bar(x=tool_distribution.index,
                           y=tool_distribution.values,
                           title="Tool Calls by Type",
                           labels={'x': 'Tool Type', 'y': 'Count'})
                st.plotly_chart(fig)
        
        # Conversation viewer
        st.header("Conversation Viewer")
        for msg in st.session_state.messages:
            with st.chat_message(msg.get('role', 'user')):
                if msg.get('name'):
                    st.caption(f"Name: {msg['name']}")
                
                # Display message content
                if msg.get('role') != 'tool':
                    if msg.get('content'):
                        st.write(normalize_text(msg['content']))
                
                # Display tool calls if present
                if msg.get('tool_calls'):
                    with st.expander("Tool Calls"):
                        for tool_call in msg['tool_calls']:
                            st.code(json.dumps(tool_call, indent=2), language='json')
                
                # Display tool responses if present
                if msg.get('tool_responses'):
                    with st.expander("Tool Responses"):
                        for tool_response in msg['tool_responses']:
                            # Normalize the content field in tool responses
                            if 'content' in tool_response:
                                # Use text display instead of write to avoid markdown interpretation
                                display_tool_content(normalize_text(tool_response['content']))
                            # Display the raw response data
                            st.code(json.dumps(tool_response, indent=2), language='json')

if __name__ == "__main__":
    main()