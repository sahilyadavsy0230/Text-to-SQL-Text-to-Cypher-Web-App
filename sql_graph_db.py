import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from neo4j import GraphDatabase

# =====================
# Env & LLM
# =====================
load_dotenv()

groq_api_key = os.getenv("groq_api_key")
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key, max_tokens=1024)

# =====================
# Session State
# =====================
if "db" not in st.session_state:
    st.session_state.db = None  # SQL connection

if "neo4j_db" not in st.session_state:
    st.session_state.neo4j_db = None  # Neo4j connection

if "db_type" not in st.session_state:
    st.session_state.db_type = "SQL"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=(
            "Hello! I am your database assistant. "
            "I can help you with SQL (MySQL) and Neo4j (Cypher). "
            "Please connect to a database to start chatting."
        ))
    ]

# =====================
# Neo4j Connection Class
# =====================
class Neo4jDBConnection:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def execute_query(self, cypher_query: str):
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            return {"error": str(e)}

    def get_schema_info(self):
        """Return labels, relationship types, and property keys for guidance."""
        schema_info = {}
        try:
            with self.driver.session(database=self.database) as session:
                labels = session.run("CALL db.labels()").values()
                rel_types = session.run("CALL db.relationshipTypes()").values()
                props = session.run("CALL db.propertyKeys()").values()

                schema_info["labels"] = [l[0] for l in labels]
                schema_info["relationship_types"] = [r[0] for r in rel_types]
                schema_info["properties"] = [p[0] for p in props]
            return schema_info
        except Exception as e:
            return {"error": str(e)}

# =====================
# Connection Helpers
# =====================

def init_sql_database(user: str, password: str, host: str, port: str, database: str):
    password_enc = quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{user}:{password_enc}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def init_neo4j_database(uri: str, user: str, password: str, database: str = "neo4j"):
    return Neo4jDBConnection(uri, user, password, database)

# =====================
# NL -> SQL Chain
# =====================

def get_sql_chain(db: SQLDatabase, user_query: str, chat_history):
    text_to_sql_prompt = """
You are an expert in translating natural language questions into syntactically correct SQL queries 
for a MySQL database.

Rules you MUST follow:
1) Use ONLY the tables and columns provided in the schema.
2) Return ONLY the SQL query — no explanations, no extra text.
3) If the question implies data modification (INSERT/UPDATE/DELETE), ensure valid syntax.
4) Maintain continuity using the given chat history if it contains context.
5) Always terminate statements with a semicolon.

SCHEMA:
{schema}

CHAT HISTORY:
{chat_history}

User Question: {question}

Respond with ONLY a SQL query.
"""
    prompt = ChatPromptTemplate.from_template(text_to_sql_prompt)
    model_instance = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it", max_tokens=1024)

    def get_schema(_):
        return db.get_table_info()

    chain = (RunnablePassthrough.assign(schema=get_schema) | prompt | model_instance | StrOutputParser())
    return chain.invoke({"question": user_query, "chat_history": chat_history})


# =====================
# NL -> Cypher Chain
# =====================

def get_cypher_chain(neo4j_db: Neo4jDBConnection, user_query: str, chat_history):
    text_to_cypher_prompt = """
You are an expert Cypher translator.

Rules:
1. Translate the user’s question into a valid Cypher query.
2. ONLY use real Cypher syntax — never invent functions.
   - To count relationships: MATCH ()-[r]->() RETURN count(r);
   - To list labels: CALL db.labels();
   - To list relationship types: CALL db.relationshipTypes();
3. Respond with only the Cypher query, no explanations, no markdown.
4. Output must be executable in Neo4j as-is.

SCHEMA:
{schema}

CHAT HISTORY:
{chat_history}

User Question: {question}

Respond with ONLY a Cypher query.
"""
    prompt = ChatPromptTemplate.from_template(text_to_cypher_prompt)
    model_instance = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it", max_tokens=1024)

    def get_schema(_):
        return str(neo4j_db.get_schema_info())

    chain = (RunnablePassthrough.assign(schema=get_schema) | prompt | model_instance | StrOutputParser())
    return chain.invoke({"question": user_query, "chat_history": chat_history})


# =====================
# SQL Result -> NL Answer
# =====================

def get_sql_response(chat_history, db: SQLDatabase, user_query: str, model: ChatGroq, sql_query: str):
    sql_result_to_nl_prompt = ChatPromptTemplate.from_template(
        """
You are a data analyst assistant.
Explain SQL database query results in clear, concise natural language.

Follow these rules:
1) Use the user question and query result to generate the answer.
2) Use the chat history for continuity.
3) Be factual; do not guess beyond the data.
4) Do NOT include the SQL query in the response.

SCHEMA:
{schema}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

SQL QUERY:
{sql_query}

SQL RESULT:
{sql_result}

Respond with a natural language answer based only on the given result.
"""
    )

    try:
        sql_result = db.run(sql_query)
        schema = db.get_table_info()
        chain = sql_result_to_nl_prompt | model | StrOutputParser()
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
            "schema": schema,
            "sql_result": sql_result,
            "sql_query": sql_query,
        })
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"


# =====================
# Cypher Result -> NL Answer
# =====================

def get_cypher_response(chat_history, neo4j_db: Neo4jDBConnection, user_query: str, model: ChatGroq, cypher_query: str):
    cypher_result_to_nl_prompt = ChatPromptTemplate.from_template(
        """
You are a data analyst assistant specialized in property graph databases (Neo4j).
Explain Cypher query results in natural language.

Rules:
1) Use the user question and query result.
2) Do NOT include the Cypher query in the response.
3) Explain relationships clearly when relevant.

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

CYPHER QUERY:
{cypher_query}

CYPHER RESULT:
{cypher_result}

Respond with a natural language answer to the user question.
"""
    )

    try:
        cypher_result = neo4j_db.execute_query(cypher_query)
        chain = cypher_result_to_nl_prompt | model | StrOutputParser()
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
            "cypher_result": cypher_result,
            "cypher_query": cypher_query,
        })
    except Exception as e:
        return f"Error executing Cypher query: {str(e)}"


# =====================
# UI: History
# =====================
for messages in st.session_state.chat_history:
    if isinstance(messages, AIMessage):
        with st.chat_message("AI"):
            st.markdown(messages.content)
    elif isinstance(messages, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(messages.content)

# =====================
# UI: Page
# =====================
st.set_page_config(page_title="Chat with Databases (SQL + Neo4j)", page_icon=":speech_balloon:")
st.title("Chat with SQL and Neo4j Databases")

# =====================
# Sidebar: Connections
# =====================
with st.sidebar:
    st.subheader("Database Settings")

    db_type = st.selectbox(
        "Select Database Type",
        ["SQL", "Neo4j"],
        index=0 if st.session_state.db_type == "SQL" else 1,
    )
    st.session_state.db_type = db_type

    if db_type == "SQL":
        st.write("Connect to SQL (MySQL)")
        st.text_input("Host", value="localhost", key="sql_host")
        st.text_input("User", value="root", key="sql_user")
        st.text_input("Password", type="password", key="sql_password")
        st.text_input("Port", value="3306", key="sql_port")
        st.text_input("Database Name", value="sql_db", key="sql_database_name")

        if st.button("Connect to SQL Database"):
            with st.spinner("Connecting to SQL database..."):
                try:
                    db = init_sql_database(
                        st.session_state["sql_user"],
                        st.session_state["sql_password"],
                        st.session_state["sql_host"],
                        st.session_state["sql_port"],
                        st.session_state["sql_database_name"],
                    )
                    st.session_state.db = db
                    st.session_state.neo4j_db = None
                    st.success("Connected to SQL database successfully!")
                except Exception as e:
                    st.error(f"Failed to connect to SQL database: {str(e)}")

    else:
        st.write("Connect to Neo4j")
        st.text_input("Neo4j URI", value="bolt://localhost:7687", key="neo4j_uri")
        st.text_input("User", value="neo4j", key="neo4j_user")
        st.text_input("Password", type="password", key="neo4j_password")
        st.text_input("Database (default: neo4j)", value="neo4j", key="neo4j_database")

        if st.button("Connect to Neo4j Database"):
            with st.spinner("Connecting to Neo4j..."):
                try:
                    neo4j_db = init_neo4j_database(
                        st.session_state["neo4j_uri"],
                        st.session_state["neo4j_user"],
                        st.session_state["neo4j_password"],
                        st.session_state["neo4j_database"],
                    )
                    test_result = neo4j_db.get_schema_info()
                    if "error" not in test_result:
                        st.session_state.neo4j_db = neo4j_db
                        st.session_state.db = None
                        st.success("Connected to Neo4j database successfully!")
                    else:
                        st.error(f"Failed to connect: {test_result['error']}")
                except Exception as e:
                    st.error(f"Failed to connect to Neo4j: {str(e)}")

# =====================
# Status Banner
# =====================
if st.session_state.db:
    st.info("✅ Connected to SQL Database")
elif st.session_state.neo4j_db:
    st.info("✅ Connected to Neo4j Database")
else:
    st.warning("⚠️ No database connected. Please connect to a database first.")

# =====================
# Chat Input
# =====================
user_query = st.chat_input("Ask me anything about your database...")

# =====================
# Chat Processing
# =====================
if user_query is not None and user_query.strip() != "":
    # Keep recent history to avoid overflow
    if len(st.session_state.chat_history) > 6:
        st.session_state.chat_history = st.session_state.chat_history[-6:]

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if st.session_state.db:  # SQL
            try:
                sql_query = get_sql_chain(st.session_state.db, user_query, st.session_state.chat_history[-5:])
                response = get_sql_response(
                    st.session_state.chat_history[-5:],
                    st.session_state.db,
                    user_query,
                    model,
                    sql_query,
                )
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                error_msg = f"Error processing SQL query: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))

        elif st.session_state.neo4j_db:  # Neo4j
            try:
                cypher_query = get_cypher_chain(st.session_state.neo4j_db, user_query, st.session_state.chat_history[-5:])
                response = get_cypher_response(
                    st.session_state.chat_history[-5:],
                    st.session_state.neo4j_db,
                    user_query,
                    model,
                    cypher_query,
                )
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                error_msg = f"Error processing Cypher query: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))
        else:
            no_conn = "Please connect to a database first (SQL or Neo4j) to start querying."
            st.warning(no_conn)
            st.session_state.chat_history.append(AIMessage(content=no_conn))

# =====================
# Help & Requirements
# =====================
with st.expander("ℹ️ How to use this application"):
    st.markdown(
        """
### SQL Database
- Connect to MySQL databases
- Ask questions in natural language (e.g., "Show all customers", "How many orders last month?")

### Neo4j Database
- Connect via bolt URI (e.g., `bolt://localhost:7687`)
- Ask questions in natural language; the app generates Cypher

### Tips
- Make sure your database service is running and network-accessible
- Use the sidebar to switch between SQL and Neo4j
        """
    )

# with st.expander("📦 Required Dependencies"):
#     st.code(
#         """
# # Install these dependencies:
# pip install streamlit
# pip install langchain
# pip install langchain-community
# pip install langchain-groq
# pip install python-dotenv
# pip install mysql-connector-python
# pip install neo4j
#         """,
#         language="bash",
#     )
