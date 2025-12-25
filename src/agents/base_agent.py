# src/agents/base_agent.py

# Imports for Agent Core
import asyncio
from typing import List, Any, Optional, Dict

# LangChain/LangSmith Imports
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage
from src.agents.response import AgentResponse, ChartArtifact
import json

async def a_load_all_mcp_tools(mcp_servers, LOGGER) -> tuple[List[Any], MultiServerMCPClient]:
    """Initializes the MCP client for ALL configured servers using HTTP/SSE transport."""
    
    LOGGER.info("🔌 Initializing Multi-Server MCP Client...")
    
    # Build the configuration for all servers
    sse_config = {}
    for server_name, server_info in mcp_servers.items():
        LOGGER.info(f"   Configuring: {server_name} ({server_info['description']})")
        LOGGER.info(f"   URL: {server_info['url']}")
        sse_config[server_name] = {
            "transport": "sse",
            "url": server_info["url"]
        }
    
    client = MultiServerMCPClient(sse_config)
    tools = await client.get_tools()
    
    LOGGER.info("✅ MCP Client initialized successfully.")
    LOGGER.info(f"📊 Total tools loaded: {len(tools)}")
    
    for tool in tools:
       LOGGER.info(f"{tool.name}: {tool.description[:60]}...")

    return tools, client

class BaseAgent:
    """
    Enhanced LangChain ReAct BaseAgent for financial tasks and chart generation.
    """
    def __init_subclass__(cls, **kwargs):
        """
        Called automatically when a subclass is created.
        Gives each subclass its own invocation counter.
        """
        super().__init_subclass__(**kwargs)
        cls._invocation_count = 0  # Each subclass gets its own counter

    def __init__(
        self,
        agent_name: str,
        llm,
        system_prompt: str,
        logger,
        mcp_servers: Optional[Dict[str, Dict]] = None,
        debug = False,
        **kwargs
    ):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent (e.g., "PortfolioAgent")
            llm: Language model instance
            system_prompt: System prompt for the agent
            logger: logger 
            mcp_servers: Dict of MCP server configs to load tools from
            debug: get debug output from agent
            **kwargs: Additional arguments for subclasses
        """

        self.agent_name = agent_name
        self.LOGGER = logger
        self.mcp_client: MultiServerMCPClient | None = None
        self.tools: List[Any] = []
        self.instance_id = id(self)  # Unique instance identifier

        # Initialize MCP tools from ALL servers
        if mcp_servers:
            try:
                self.tools, self.mcp_client = asyncio.run(a_load_all_mcp_tools(mcp_servers, self.LOGGER))
                self.LOGGER.info(f"✅ Successfully loaded {len(self.tools)} tools from all MCP servers")
            except Exception as e:
                self.LOGGER.error(f"FATAL ERROR: Could not connect to MCP servers: {e}")
                self.tools = []
                import traceback
                self.LOGGER.error(traceback.format_exc())
                self.tools = []
        else:
            self.LOGGER.info(f"No MCP servers specified for {agent_name}") 

        # Create the core ReAct agent chain
        self.core_agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=debug,
        )
        
        self.LOGGER.debug(f"✅ {self.agent_name} initialized with {len(self.tools)} tools (Instance ID: {self.instance_id})")


    async def run_query(self, history: List[BaseMessage], session_id: str, portfolio: Optional[Dict[str, float]] = None) -> AgentResponse:
        """
        Runs the agent against the conversation history and returns the response.
        Handles multiple sequential or parallel tool calls.
        """
        self.LOGGER.info(f"Processing query: {history[-1].content[:50]}...")

        self.__class__._invocation_count += 1
        current_invocation = self.__class__._invocation_count
        
        self.LOGGER.debug(f"🤖 {self.agent_name} AGENT - Query #{current_invocation}")
        self.LOGGER.debug(f"🆔 Instance ID: {self.instance_id}")
        self.LOGGER.debug(f"📝 Session ID: {session_id}")
        self.LOGGER.debug(f"💬 User Query: {history[-1].content[:100]}...")
        self.LOGGER.debug(f"📊 History Length: {len(history)} messages")

        # Log the last few messages for context
        self.LOGGER.debug(f"📜 Message History:")
        for i, msg in enumerate(history[-3:], start=max(0, len(history)-3)):
            msg_type = "👤 USER" if isinstance(msg, HumanMessage) else "🤖 AI"
            content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            self.LOGGER.debug(f"   [{i}] {msg_type}: {content_preview}")

        tool_call_count = 0
        tool_call_details = []
        generated_charts = []
        updated_portfolio = None

        try:
            working_history = list(history)
            max_iterations = 15  # ✅ Increased for complex operations
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                self.LOGGER.debug(f"\n{'='*60}")
                self.LOGGER.debug(f"🔄 ITERATION {iteration}")
                self.LOGGER.debug(f"{'='*60}")
                self.LOGGER.debug(f"📝 Working history length: {len(working_history)} messages")
                
                # Invoke the agent
                response = await self.core_agent.ainvoke(
                    {"messages": working_history}
                )
                
                if not isinstance(response, dict) or "messages" not in response:
                    self.LOGGER.warning(f"⚠️  Unexpected response structure: {type(response)}")
                    break
                
                new_messages = response["messages"]
                self.LOGGER.debug(f"📬 Received {len(new_messages)} new message(s)")
                
                # Log each new message
                for i, msg in enumerate(new_messages):
                    msg_type = type(msg).__name__
                    self.LOGGER.debug(f"  [{i}] {msg_type}")
                    
                    if msg_type == "AIMessage":
                        # ✅ Log AIMessage details
                        has_content = hasattr(msg, 'content') and msg.content
                        has_tools = hasattr(msg, 'tool_calls') and msg.tool_calls
                        self.LOGGER.debug(f"      AIMessage - has_content: {has_content}, has_tool_calls: {has_tools}")
                        
                        if has_tools:
                            self.LOGGER.debug(f"      🔧 Contains {len(msg.tool_calls)} tool call(s):")
                            for tc in msg.tool_calls:
                                tool_name = tc.get('name', 'unknown')
                                self.LOGGER.debug(f"         - {tool_name}")
                                tool_call_details.append(tool_name)
                                tool_call_count += 1
                    
                    elif msg_type == "ToolMessage":
                        tool_name = getattr(msg, 'name', 'unknown')
                        self.LOGGER.debug(f"      ✅ Tool result for: {tool_name}")

                        # Extract charts if applicable
                        if 'chart' in tool_name.lower() and hasattr(msg, 'content'):
                            try:
                                for chart_data in msg.content:
                                    if isinstance(chart_data, dict) and chart_data.get('type') == 'text':
                                        chart = json.loads(chart_data['text'])
                                        chart_artifact = ChartArtifact(
                                            title=chart['title'],
                                            filename=f"{chart['filename']}"
                                        )
                                        generated_charts.append(chart_artifact)
                                        self.LOGGER.info(f"         📊 Chart captured: {chart_artifact.title}")
                            except Exception as e:
                                self.LOGGER.warning(f"         ⚠️  Could not parse chart data: {e}")

                        elif 'portfolio' in tool_name.lower() and hasattr(msg, 'content'):
                            try:
                                # Portfolio tools return dict directly
                                if isinstance(msg.content, dict):
                                    updated_portfolio = msg.content
                                    self.LOGGER.info(f"         💼 Portfolio updated: Total ${sum(updated_portfolio.values()):,.0f}")
                                # Or if wrapped in content array
                                elif isinstance(msg.content, list):
                                    for content_item in msg.content:
                                        if isinstance(content_item, dict):
                                            if 'Equities' in content_item:  # It's a portfolio
                                                updated_portfolio = content_item
                                                self.LOGGER.info(f"         💼 Portfolio updated: Total ${sum(updated_portfolio.values()):,.0f}")
                            except Exception as e:
                                self.LOGGER.warning(f"         ⚠️  Could not extract portfolio: {e}")
                
                # ✅ Log total charts captured
                self.LOGGER.debug(f"      📋 Total charts captured so far: {len(generated_charts)}")
                
                # Check if the last message has tool calls
                last_message = new_messages[-1]
                has_tool_calls = (
                    hasattr(last_message, 'tool_calls') and 
                    last_message.tool_calls and 
                    len(last_message.tool_calls) > 0
                )
                
                # ✅ Log details about the last message
                last_msg_type = type(last_message).__name__
                self.LOGGER.debug(f"📋 Last message type: {last_msg_type}")
                if hasattr(last_message, 'content'):
                    has_content = bool(last_message.content)
                    self.LOGGER.debug(f"   - has_content: {has_content}")
                self.LOGGER.debug(f"   - has_tool_calls: {has_tool_calls}")
                
                if has_tool_calls:
                    self.LOGGER.debug(f"➡️  Agent needs to execute tools, continuing loop...")
                    
                    # ✅ Debug logging before extend
                    self.LOGGER.debug(f"📝 About to extend history:")
                    self.LOGGER.debug(f"   Current history length: {len(working_history)}")
                    self.LOGGER.debug(f"   New messages count: {len(new_messages)}")
                    
                    # Add new messages to history for next iteration
                    working_history.extend(new_messages)
                    
                    self.LOGGER.debug(f"   History after extend: {len(working_history)} messages")
                else:
                    # Agent is done - final response received
                    self.LOGGER.info(f"🎯 Agent has no more tool calls - completing")
                    self.LOGGER.debug(f"\n{'='*60}")
                    self.LOGGER.debug(f"✅ AGENT COMPLETE")
                    self.LOGGER.debug(f"{'='*60}")
                    self.LOGGER.debug(f"🔧 Total tool calls: {tool_call_count}")
                    self.LOGGER.debug(f"📊 Charts generated: {len(generated_charts)}")
                    
                    if tool_call_details:
                        self.LOGGER.debug(f"🔨 Tool execution sequence:")
                        for i, tool in enumerate(tool_call_details, 1):
                            self.LOGGER.debug(f"   {i}. {tool}")
                    
                    # Return final response
                    if hasattr(last_message, 'content'):
                        response_preview = last_message.content[:150] + "..." if len(last_message.content) > 150 else last_message.content
                        self.LOGGER.debug(f"💬 Response Preview: {response_preview}")
                        
                        return AgentResponse(
                            agent=self.agent_name,
                            message=last_message.content,
                            charts=generated_charts,
                            portfolio=updated_portfolio
                        )
                    else:
                        return AgentResponse(
                            agent=self.agent_name,
                            message=str(last_message),
                            charts=generated_charts,
                            portfolio=updated_portfolio
                        )
            
            # Max iterations reached
            self.LOGGER.warning(f"⚠️  Reached max iterations ({max_iterations})")
            self.LOGGER.warning(f"Tool calls made: {tool_call_count}, Tools: {tool_call_details}")
            
            # ✅ Return partial results
            return AgentResponse(
                agent=self.agent_name,
                message="I apologize, but I couldn't complete the request within the iteration limit. Here are the results I was able to generate.",
                charts=generated_charts,
                portfolio=updated_portfolio
            )
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.LOGGER.error(f"❌ Error: {error_msg}")
            import traceback
            self.LOGGER.error(traceback.format_exc())
            
            # ✅ Return partial results even on error
            self.LOGGER.info(f"📊 Returning partial results despite error: {len(generated_charts)} charts, portfolio: {updated_portfolio is not None}")
            
            return AgentResponse(
                agent=self.agent_name,
                message=f"I encountered an error, but here are the results I was able to generate before the error occurred.",
                charts=generated_charts,
                portfolio=updated_portfolio
            )

    async def cleanup(self):
        """Cleanup method to properly close the MCP client connection."""
        if self.mcp_client:
            try:
                # Note: Check if your MCP client has a close/cleanup method
                # await self.mcp_client.close()
                self.LOGGER.info("🧹 Cleanup complete")
            except Exception as e:
                self.LOGGER.error(f"Error during cleanup: {e}")