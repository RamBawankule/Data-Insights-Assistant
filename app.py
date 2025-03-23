import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genAI
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
import time
import io
import tempfile

# App configuration
st.set_page_config(
    page_title="Data Insights Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'db_schema' not in st.session_state:
    st.session_state.db_schema = {}
if 'db_type' not in st.session_state:
    st.session_state.db_type = None
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'run_query' not in st.session_state:
    st.session_state.run_query = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'sample_questions' not in st.session_state:
    st.session_state.sample_questions = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'model_name' not in st.session_state:
    st.session_state.model_name = "gemini-2.0-flash"

# Initial setup dialog for API key and model selection
if not st.session_state.api_key_set:
    st.title("Data Insights Assistant - Setup")
    st.markdown("Welcome to the Data Insights Assistant! Please provide your API key and select a model to continue.")
    
    # Model selection
    model_options = ["gemini-2.0-flash", "gemini-1.5-pro"]
    selected_model = st.selectbox(
        "Select Gemini Model:", 
        options=model_options,
        index=0,
        help="Choose the Gemini model to use for queries. Different models have different capabilities and response speeds."
    )
    
    # API key input
    api_key_input = st.text_input(
        "Enter your Google API Key:", 
        type="password",
        help="Your Google API key for authenticating with Gemini AI. This is required to use the service."
    )
    
    # Option to load from .env file
    # use_env_file = st.checkbox("Use API key from .env file (if available)", value=True)
    
    # Submit button for configuration
    if st.button("Start Application"):
        # if use_env_file:
        #     # Try to load from .env file first
        #     load_dotenv()
        #     env_api_key = os.getenv("GOOGLE_API_KEY")
        #     if env_api_key and not api_key_input:
        #         api_key_input = env_api_key
        
        if api_key_input:
            # Configure Gemini AI
            genAI.configure(api_key=api_key_input)
            st.session_state.api_key_set = True
            st.session_state.model_name = selected_model
            st.success("Setup complete! Please wait while the application loads...")
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
        else:
            st.error("API key is required. Please enter your API key or ensure it's set in your .env file.")
    
    # Stop execution to prevent the rest of the app from loading
    st.stop()

# Cache the model to avoid recreating it for each query
@st.cache_resource
def get_model(model_name=st.session_state.model_name):
    return genAI.GenerativeModel(model_name)

# Function to create SQLite connection from different file types
def create_connection_from_file(uploaded_file):
    try:
        # Get file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext == '.db':
            # For SQLite database files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            conn = sqlite3.connect(tmp_path, check_same_thread=False)
            db_type = "sqlite"
            
            # Get schema information
            cursor = conn.cursor()
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            schema = {}
            
            for table in tables:
                table_name = table[0]
                columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
                schema[table_name] = [col[1] for col in columns]
            
            return conn, schema, db_type
            
        elif file_ext == '.csv':
            # For CSV files
            df = pd.read_csv(uploaded_file)
            
            # Create in-memory SQLite database
            conn = sqlite3.connect(':memory:', check_same_thread=False)
            df.to_sql('data', conn, index=False, if_exists='replace')
            
            schema = {'data': df.columns.tolist()}
            db_type = "csv"
            
            return conn, schema, db_type
            
        elif file_ext in ['.xlsx', '.xls']:
            # For Excel files
            df = pd.read_excel(uploaded_file, sheet_name=None)
            
            # Create in-memory SQLite database
            conn = sqlite3.connect(':memory:', check_same_thread=False)
            
            # Handle multiple sheets
            schema = {}
            for sheet_name, sheet_df in df.items():
                # Sanitize sheet name to be a valid SQL table name
                table_name = ''.join(c if c.isalnum() else '_' for c in sheet_name)
                sheet_df.to_sql(table_name, conn, index=False, if_exists='replace')
                schema[table_name] = sheet_df.columns.tolist()
                
            db_type = "excel"
            return conn, schema, db_type
            
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None, None, None
            
    except Exception as e:
        st.error(f"Error creating database connection: {e}")
        return None, None, None

# Generate sample questions based on the database schema
def generate_schema_based_questions(schema, db_type):
    try:
        schema_description = ""
        for table, columns in schema.items():
            schema_description += f"Table: {table}, Columns: {', '.join(columns)}\n"
        
        prompt = f"""
        Based on this database schema:
        {schema_description}
        
        Generate 5 natural language questions that users might ask about this data.
        The questions should be diverse and show the range of analyses possible.
        
        Format: Return exactly 5 questions, one per line, without numbering or bullet points.
        Each question should be clear and specific to this database.
        """
        
        model = get_model()
        response = model.generate_content(prompt)
        
        # Split by lines and clean
        questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        
        # Return at most 5 questions
        return questions[:5]
        
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return [
            "How many records are in each table?",
            "What's the average of numeric columns?",
            "Show me the first 10 records",
            "Count unique values in each column",
            "Summarize the data by categories"
        ]

# Cache the SQL response to avoid redundant API calls
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_gemini_response(question, schema_description):
    try:
        prompt = f"""
        Convert the following natural language question to a SQL query for a database with this schema:
        {schema_description}
        
        Return only valid SQL code without backticks, SQL keyword, or explanations.

        The SQL code should not have ``` at the beginning or end, and should not include the word "sql" in the output. No other text.
        
        Question: {question}
        """
        
        model = get_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return ""

# Fixed: Added underscore to make the connection parameter unhashable
@st.cache_data(ttl=300)  # Cache for 5 minutes
def read_sql_query(sql, _conn):
    try:
        df = pd.read_sql_query(sql, _conn)
        return df
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return pd.DataFrame()

# Optimized visualization function
def auto_visualize(df):
    """Automatically create an appropriate visualization based on the data"""
    try:
        if df.empty or (len(df) == 1 and len(df.columns) == 1):
            return None
        
        num_columns = df.select_dtypes(include=['number']).columns
        cat_columns = df.select_dtypes(exclude=['number']).columns
        
        # For small results, return simple visualizations
        if len(df.columns) <= 2:
            if len(num_columns) == 1 and len(cat_columns) == 1:
                fig = px.bar(df, x=cat_columns[0], y=num_columns[0], 
                            title=f"{num_columns[0]} by {cat_columns[0]}",
                            color_discrete_sequence=['#3366CC'])
                return fig
            elif len(num_columns) == 1 and len(cat_columns) == 0:
                fig = px.histogram(df, x=num_columns[0], 
                                title=f"Distribution of {num_columns[0]}",
                                color_discrete_sequence=['#3366CC'])
                return fig
            elif len(num_columns) == 2:
                fig = px.scatter(df, x=num_columns[0], y=num_columns[1], 
                                title=f"{num_columns[1]} vs {num_columns[0]}",
                                color_discrete_sequence=['#3366CC'])
                return fig
            elif len(cat_columns) >= 1:
                counts = df[cat_columns[0]].value_counts().reset_index()
                counts.columns = [cat_columns[0], 'count']
                fig = px.bar(counts, x=cat_columns[0], y='count', 
                            title="Count by Category",
                            color_discrete_sequence=['#3366CC'])
                return fig
        
        # For multi-column datasets with numeric and categorical columns
        elif len(num_columns) >= 1 and len(cat_columns) >= 1:
            fig = px.bar(df, x=cat_columns[0], y=num_columns[0], 
                        title=f"{num_columns[0]} by {cat_columns[0]}",
                        color_discrete_sequence=['#3366CC'])
            return fig
            
        # Limit data points for larger datasets to improve performance
        elif len(df) > 100 and len(num_columns) >= 1:
            # Sample the data to avoid overplotting
            sample_size = min(100, len(df))
            sampled_df = df.sample(sample_size)
            numeric_col = num_columns[0]
            
            if len(cat_columns) >= 1:
                fig = px.box(sampled_df, x=cat_columns[0], y=numeric_col, 
                            title=f"Distribution of {numeric_col} by {cat_columns[0]}",
                            color_discrete_sequence=['#3366CC'])
            else:
                fig = px.line(sampled_df, y=numeric_col, 
                            title=f"Trend of {numeric_col}",
                            color_discrete_sequence=['#3366CC'])
            return fig
            
        return None
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

# Fixed: Using string sample instead of dataframe to avoid unhashable issues
@st.cache_data(ttl=600)  # Cache for 10 minutes
def generate_formatted_insights(query, sample_string, total_rows, schema_description):
    """Generate insights using fewer tokens and more specific instructions"""
    try:
        if not sample_string:
            return []
        
        # Updated prompt with clearer formatting instructions
        insights_prompt = f"""
        Data analysis task: Generate 3 brief business insights for this query: "{query}"
        
        Database schema:
        {schema_description}
        
        Data sample (showing portion of {total_rows} total rows):
        {sample_string}
        
        Format: Return exactly 3 separate insights, one per line WITHOUT bullet points or numbering.
        Each should identify one pattern in the data and suggest one specific action. Maximum 20 words per insight.
        """
        
        model = get_model()
        response = model.generate_content(insights_prompt)
        
        # Split and clean insights
        insights_list = [insight.strip() for insight in response.text.strip().split('\n') if insight.strip()]
        
        # Remove any leading asterisks, numbers, or other markers
        clean_insights = []
        for insight in insights_list[:3]:
            # Remove any leading asterisks, numbers, dashes, etc.
            clean_insight = insight
            for prefix in ['* ', '- ', '• ', '1. ', '2. ', '3. ', '1) ', '2) ', '3) ']:
                if clean_insight.startswith(prefix):
                    clean_insight = clean_insight[len(prefix):]
            clean_insights.append(clean_insight)
            
        return clean_insights
    except Exception as e:
        return [f"Could not generate insights: {str(e)[:50]}..."]

# Main app section
st.title("Data Insights Assistant")
st.markdown("Ask questions in natural language and get SQL queries, visualizations, and insights!")

# Sidebar configuration
with st.sidebar:
    st.title("📊 Data Assistant")
    
    # Model info display
    st.info(f"Using model: **{st.session_state.model_name}**")
    if st.button("Change API Key or Model"):
        st.session_state.api_key_set = False
        st.rerun()
    
    # Move database upload to sidebar
    st.subheader("Database Upload")
    uploaded_file = st.file_uploader(
        "Upload a database file", 
        type=["db", "csv", "xlsx", "xls"],
        help="Select a .db file for SQLite databases, .csv for CSV files, or .xlsx/.xls for Excel files"
    )
    
    # Show supported file formats
    with st.expander("Supported File Formats"):
        st.markdown("""
        - **SQLite Database (.db)**
        - **CSV Files (.csv)**
        - **Excel Files (.xlsx, .xls)**
        """)
    
    # Process uploaded file
    if uploaded_file and (st.session_state.current_file != uploaded_file.name):
        st.session_state.current_file = uploaded_file.name
        
        with st.spinner("Processing database file..."):
            # Create connection based on file type
            conn, schema, db_type = create_connection_from_file(uploaded_file)
            
            if conn and schema:
                st.session_state.db_connection = conn
                st.session_state.db_schema = schema
                st.session_state.db_type = db_type
                
                # Generate sample questions based on schema
                st.session_state.sample_questions = generate_schema_based_questions(schema, db_type)
                
                st.success(f"Loaded {db_type.upper()}: {uploaded_file.name}")
            else:
                st.error("Failed to process the file.")
    
    # Display sample questions
    if st.session_state.sample_questions:
        st.markdown("### Suggested Questions")
        st.write("Click to try these examples:")
        
        for question in st.session_state.sample_questions:
            if st.button(question):
                st.session_state.user_query = question
                st.session_state.run_query = True
    else:
        if not st.session_state.db_connection:
            st.info("Upload a database file to see suggested questions.")

# Display database schema if available
if st.session_state.db_schema:
    with st.expander("Database Schema", expanded=False):
        for table, columns in st.session_state.db_schema.items():
            st.markdown(f"**Table: {table}**")
            st.write(f"Columns: {', '.join(columns)}")
            st.divider()

if st.session_state.db_connection:
    col1, col2 = st.columns([0.85, 0.15], gap="small")
    
    with col1:
        user_query = st.text_input(
            "Enter your question about the database:",
            value=st.session_state.get('user_query', ''),
            key="query_input"
        )
        st.session_state.user_query = user_query

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space
        submit_button = st.button("Run Query", use_container_width=True)

        if submit_button:
            st.session_state.run_query = True
            # st.success("Query submitted!")

    # Generate schema description for prompts
    schema_description = ""
    for table, columns in st.session_state.db_schema.items():
        schema_description += f"Table: {table}, Columns: {', '.join(columns)}\n"

    # Execute when query is submitted
    if st.session_state.run_query and st.session_state.user_query:
        # Reset the flag
        st.session_state.run_query = False
        
        with st.spinner("Generating SQL query..."):
            # Get SQL query
            sql_query = get_gemini_response(st.session_state.user_query, schema_description)
            
            # Add to history
            if sql_query:
                st.session_state.history.append({
                    "question": st.session_state.user_query,
                    "sql": sql_query,
                    "timestamp": time.strftime("%H:%M:%S")
                })
        
        # If we got a valid SQL query
        if sql_query:
            # Display the query with better formatting
            with st.expander("Generated SQL Query", expanded=True):
                st.code(sql_query, language="sql")
            
            # Execute the query with timeout protection
            with st.spinner("Executing query..."):
                conn = st.session_state.db_connection
                df = read_sql_query(sql_query, conn)
            
            # Display results in tabs for better organization
            if not df.empty:
                tabs = st.tabs(["Data Results", "Visualization", "Insights"])
                
                with tabs[0]:
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        label="Download Results CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"query_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                    )
                
                with tabs[1]:
                    fig = auto_visualize(df)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No appropriate visualization available for this query result.")
                
                with tabs[2]:
                    # Use sample of data to reduce tokens sent to API
                    # Convert to string to avoid unhashable objects in caching
                    sample_size = min(5, len(df))
                    sample_df = df.head(sample_size)
                    sample_string = sample_df.to_string(index=False)
                    
                    with st.spinner("Generating insights..."):
                        insights = generate_formatted_insights(
                            st.session_state.user_query, 
                            sample_string, 
                            len(df),
                            schema_description
                        )
                    
                    # Display insights with proper numbering in Streamlit
                    for i, insight in enumerate(insights):
                        st.markdown(f"{i+1}. {insight}")
            else:
                st.warning("No results found for this query.")

    # Query history section
    if st.session_state.history:
        with st.expander("Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 queries
                st.markdown(f"**{item['timestamp']}**: {item['question']}")
                st.code(item['sql'], language="sql")
                if i < len(st.session_state.history) - 1:
                    st.divider()
else:
    # Show instructions when no database is connected
    st.info("👈 Please upload a database file in the sidebar to get started.")

# Footer with clean design
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("💡 **Tip**: After uploading your database, try the suggested questions or ask your own!")
with col2:
    st.markdown("**Created by Ram Bawankule😊**")