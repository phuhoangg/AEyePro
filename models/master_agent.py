import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from models.recommend_agent import RecommendAgent
from models.retrieval_agent import RetrievalAgent
from core.health_data_collector import HealthDataCollector

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Định nghĩa các loại sub-agent"""
    RECOMMEND = "recommend_agent"
    RETRIEVAL = "retrieval_agent"
    NONE = "none"
    GENERAL_CHAT = "general_chat"  # Thêm loại chat chung


@dataclass
class AgentResponse:
    """Cấu trúc phản hồi từ sub-agent"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    agent_type: Optional[AgentType] = None


@dataclass
class UserQuery:
    """Cấu trúc câu hỏi từ người dùng"""
    text: str
    timestamp: datetime
    intent: Optional[str] = None
    entities: Optional[List[str]] = None


class SmartIntentClassifier:
    """Phân loại ý định thông minh với khả năng nhận biết câu hỏi không liên quan"""

    def __init__(self):
        # Từ khóa chính xác cho recommend_agent
        self.recommend_keywords = [
            "cảnh báo", "alert", "khuyến nghị", "recommend", "hiện tại", "bây giờ",
            "real-time", "thời gian thực", "tư thế", "posture", "mắt", "eye",
            "nghỉ", "break", "ngồi", "sitting", "làm việc", "work", "sức khỏe",
            "health", "mỏi", "tired", "stress", "căng thẳng", "monitor", "theo dõi",
            "start", "stop", "bắt đầu", "dừng", "status", "trạng thái"
        ]

        # Từ khóa chính xác cho retrieval_agent
        self.retrieval_keywords = [
            "phân tích", "analysis", "xu hướng", "trend", "lịch sử", "history",
            "thống kê", "statistic", "báo cáo", "report", "so sánh", "compare",
            "tuần", "week", "tháng", "month", "ngày", "day", "thời gian", "time",
            "hiệu suất", "performance", "thói quen", "habit", "pattern", "mẫu",
            "dữ liệu", "data", "tìm kiếm", "search", "truy vấn", "query",
            "buổi sáng", "morning", "buổi tối", "evening", "buổi chiều", "afternoon"
        ]

        # Từ khóa cho cả hai agent
        self.both_keywords = [
            "tổng hợp", "overview", "tổng quan", "general", "toàn bộ", "all",
            "comprehensive", "chi tiết", "detailed", "full", "đầy đủ"
        ]

        # Từ khóa liên quan đến sức khỏe/làm việc - chỉ những câu có từ này mới cần sub-agent
        self.health_work_context = [
            "sức khỏe", "health", "làm việc", "work", "ngồi", "sitting",
            "máy tính", "computer", "văn phòng", "office", "mắt", "eye",
            "lưng", "back", "cổ", "neck", "tư thế", "posture", "mỏi", "tired",
            "đau", "pain", "nghỉ", "break", "giải lao", "exercise", "thể dục",
            "monitor", "theo dõi", "tracking", "productivity", "năng suất"
        ]

        # Từ khóa chat chung - không cần sub-agent
        self.general_chat_keywords = [
            "xin chào", "hello", "hi", "chào", "cảm ơn", "thank", "tạm biệt", "bye",
            "thời tiết", "weather", "ăn", "eat", "uống", "drink", "phim", "movie",
            "nhạc", "music", "tin tức", "news", "bóng đá", "football", "game",
            "học", "study", "trường", "school", "bạn", "friend", "gia đình", "family",
            "yêu", "love", "vui", "happy", "buồn", "sad", "giải trí", "entertainment",
            "du lịch", "travel", "mua sắm", "shopping", "nấu ăn", "cooking"
        ]

        # Các câu hỏi thông thường không cần sub-agent
        self.general_patterns = [
            r".*là gì.*",
            r".*how to.*",
            r".*cách.*",
            r".*why.*",
            r".*tại sao.*",
            r".*định nghĩa.*",
            r".*giải thích.*",
            r".*explain.*",
            r".*what.*",
            r".*where.*",
            r".*when.*",
            r".*who.*"
        ]

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Phân loại ý định thông minh"""
        query_lower = query.lower()

        # Bước 1: Kiểm tra ngữ cảnh sức khỏe/công việc
        has_health_context = any(keyword in query_lower for keyword in self.health_work_context)

        # Bước 2: Kiểm tra câu hỏi chat chung
        has_general_chat = any(keyword in query_lower for keyword in self.general_chat_keywords)

        # Bước 3: Kiểm tra pattern câu hỏi chung
        is_general_question = any(re.search(pattern, query_lower) for pattern in self.general_patterns)

        # Bước 4: Tính điểm cho các agent
        recommend_score = sum(1 for keyword in self.recommend_keywords if keyword in query_lower)
        retrieval_score = sum(1 for keyword in self.retrieval_keywords if keyword in query_lower)
        both_score = sum(1 for keyword in self.both_keywords if keyword in query_lower)

        result = {
            "primary_agent": AgentType.NONE,
            "secondary_agent": None,
            "confidence": 0.0,
            "reasoning": "",
            "use_both": False,
            "use_general_chat": False,
            "has_health_context": has_health_context
        }

        # Logic quyết định
        if has_general_chat and not has_health_context:
            result["use_general_chat"] = True
            result["primary_agent"] = AgentType.GENERAL_CHAT
            result["confidence"] = 0.9
            result["reasoning"] = "Câu hỏi chat chung, không cần sử dụng sub-agent"

        elif is_general_question and not has_health_context:
            result["use_general_chat"] = True
            result["primary_agent"] = AgentType.GENERAL_CHAT
            result["confidence"] = 0.8
            result["reasoning"] = "Câu hỏi thông thường, không liên quan đến dữ liệu sức khỏe"

        elif not has_health_context and recommend_score == 0 and retrieval_score == 0:
            result["use_general_chat"] = True
            result["primary_agent"] = AgentType.GENERAL_CHAT
            result["confidence"] = 0.7
            result["reasoning"] = "Câu hỏi không liên quan đến sức khỏe/công việc"

        elif has_health_context:
            if both_score > 0 or abs(recommend_score - retrieval_score) <= 1:
                result["use_both"] = True
                result["primary_agent"] = AgentType.RECOMMEND
                result["secondary_agent"] = AgentType.RETRIEVAL
                result["confidence"] = 0.8
                result["reasoning"] = "Câu hỏi về sức khỏe cần thông tin từ cả hai agent"

            elif recommend_score > retrieval_score:
                result["primary_agent"] = AgentType.RECOMMEND
                result["confidence"] = min(recommend_score / 3.0, 1.0)
                result["reasoning"] = "Câu hỏi về cảnh báo và khuyến nghị sức khỏe"

            elif retrieval_score > recommend_score:
                result["primary_agent"] = AgentType.RETRIEVAL
                result["confidence"] = min(retrieval_score / 3.0, 1.0)
                result["reasoning"] = "Câu hỏi về phân tích dữ liệu sức khỏe"

            else:
                result["primary_agent"] = AgentType.RECOMMEND
                result["confidence"] = 0.5
                result["reasoning"] = "Câu hỏi về sức khỏe, sử dụng recommend agent mặc định"
        else:
            result["use_general_chat"] = True
            result["primary_agent"] = AgentType.GENERAL_CHAT
            result["confidence"] = 0.6
            result["reasoning"] = "Không thể xác định ý định, sử dụng chat chung"

        return result


class LLMResponseGenerator:
    """Sinh phản hồi sử dụng LLM với khả năng chat chung"""

    def __init__(self, model_path: str = "models/Llama-3.2-3B-Instruct-Q8_0.gguf"):
        """Khởi tạo LLM"""
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=-1,
                temperature=0.7,
                max_tokens=512,
                n_ctx=2048,
                verbose=False
            )
            if self.llm is None:
                raise RuntimeError("LLM model not initialized. Kiểm tra lại file model hoặc cấu hình máy.")

            self.synthesis_template = PromptTemplate(
                input_variables=["query", "agent_responses", "context"],
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id>
Cutting Knowledge Date: December 2023 Today Date: 26 Jul 2024

Bạn là một AI Assistant chuyên nghiệp hỗ trợ theo dõi sức khỏe và phân tích dữ liệu. Nhiệm vụ của bạn là:

1. Tổng hợp thông tin từ các agent một cách rõ ràng và hữu ích
2. Ưu tiên cảnh báo sức khỏe quan trọng
3. Giải thích dữ liệu phân tích một cách dễ hiểu
4. Sử dụng emoji phù hợp để làm nổi bật thông tin
5. Trình bày kết quả với định dạng rõ ràng, có dấu cách và xuống dòng phù hợp

Quy tắc trình bày:
- Sử dụng **in đậm** cho tiêu đề và điểm quan trọng
- Thêm dấu cách và xuống dòng để dễ đọc
- Sử dụng emoji để phân loại thông tin (⚠️ cảnh báo, 📊 phân tích, ✅ bình thường, 🔄 hành động)
- Không lặp lại thông tin không cần thiết<|eot_id|><|start_header_id|>user<|end_header_id>

Câu hỏi người dùng: {query}

Thông tin từ các agent:
{agent_responses}

Ngữ cảnh hội thoại trước:
{context}

Hãy tổng hợp thông tin trên và trả lời câu hỏi của người dùng một cách rõ ràng, hữu ích với format dễ đọc.<|eot_id|><|start_header_id|>assistant<|end_header_id>

"""
            )

            self.general_chat_template = PromptTemplate(
                input_variables=["query", "context"],
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id>
Cutting Knowledge Date: December 2023 Today Date: 26 Jul 2024

Bạn là một AI Assistant thân thiện và hữu ích. Bạn có thể trò chuyện về nhiều chủ đề khác nhau một cách tự nhiên và vui vẻ.

Nguyên tắc trả lời:
- Trả lời một cách tự nhiên, thân thiện
- Cung cấp thông tin hữu ích và chính xác
- Sử dụng emoji phù hợp để làm cho cuộc trò chuyện sinh động hơn
- Giữ phong cách gần gũi, dễ hiểu
- Nếu không biết thông tin, hãy thành thật nói rằng không biết

Lưu ý: Bạn cũng có khả năng hỗ trợ theo dõi sức khỏe, nếu người dùng hỏi về sức khỏe hoặc công việc.<|eot_id|><|start_header_id|>user<|end_header_id>

Câu hỏi: {query}

Ngữ cảnh hội thoại trước:
{context}

Hãy trả lời câu hỏi một cách tự nhiên và hữu ích.<|eot_id|><|start_header_id|>assistant<|end_header_id>

"""
            )

            self.synthesis_chain = LLMChain(
                llm=self.llm,
                prompt=self.synthesis_template
            )

            self.general_chat_chain = LLMChain(
                llm=self.llm,
                prompt=self.general_chat_template
            )

        except Exception as e:
            logger.error(f"Lỗi khởi tạo LLM: {e}")
            raise RuntimeError(f"Không thể khởi tạo LLM: {e}")

    def generate_response(self, query: str, agent_responses: Dict = None, context: str = "",
                          use_general_chat: bool = False) -> str:
        """Sinh phản hồi từ thông tin các agent hoặc chat chung"""

        if use_general_chat:
            return self._generate_general_chat_response(query, context)

        if not agent_responses:
            return self._generate_general_chat_response(query, context)

        if not self.synthesis_chain:
            return self._fallback_response(query, agent_responses)

        try:
            agent_info = []
            for agent_type, response in agent_responses.items():
                if response.success:
                    formatted_data = self._format_agent_data(agent_type, response.data)
                    agent_info.append(f"**{agent_type}**:\n{formatted_data}")
                else:
                    agent_info.append(f"**{agent_type}**: ❌ Lỗi - {response.error}")

            agent_responses_text = "\n\n".join(agent_info)

            response = self.synthesis_chain.run(
                query=query,
                agent_responses=agent_responses_text,
                context=context if context else "Không có ngữ cảnh trước đó."
            )

            formatted_response = self._format_final_response(response.strip())
            return formatted_response

        except Exception as e:
            logger.error(f"Lỗi sinh phản hồi: {e}")
            return self._fallback_response(query, agent_responses)

    def _generate_general_chat_response(self, query: str, context: str = "") -> str:
        """Sinh phản hồi cho chat chung"""
        try:
            if not self.general_chat_chain:
                return self._fallback_general_response(query)

            response = self.general_chat_chain.run(
                query=query,
                context=context if context else "Không có ngữ cảnh trước đó."
            )

            return self._format_final_response(response.strip())

        except Exception as e:
            logger.error(f"Lỗi sinh phản hồi chat chung: {e}")
            return self._fallback_general_response(query)

    def _fallback_general_response(self, query: str) -> str:
        """Phản hồi dự phòng cho chat chung"""
        return f"🤖 Tôi hiểu bạn muốn hỏi về: '{query}'\n\nTôi sẵn sàng trò chuyện với bạn! Tuy nhiên, tôi chuyên về hỗ trợ sức khỏe và theo dõi công việc. Bạn có muốn hỏi gì về sức khỏe hoặc thói quen làm việc không? 😊"

    def _format_agent_data(self, agent_type: str, data: Dict) -> str:
        """Format dữ liệu từ agent để dễ đọc"""
        if not data:
            return "Không có dữ liệu"

        formatted_parts = []

        if agent_type == "recommend_agent":
            if data.get("action") == "recommendations":
                if "alerts" in data:
                    alerts = data["alerts"]
                    if alerts:
                        formatted_parts.append("⚠️ **Cảnh báo:**")
                        for alert in alerts:
                            formatted_parts.append(f"  - {alert}")

                if "recommendations" in data:
                    recs = data["recommendations"]
                    if recs:
                        formatted_parts.append("\n💡 **Khuyến nghị:**")
                        for rec in recs:
                            formatted_parts.append(f"  - {rec}")

                if "current_status" in data:
                    status = data["current_status"]
                    formatted_parts.append(f"\n📊 **Trạng thái hiện tại:** {status}")

            elif data.get("action") == "started":
                formatted_parts.append(f"🔄 {data.get('message', 'Đã bắt đầu theo dõi')}")

            elif data.get("action") == "stopped":
                formatted_parts.append(f"⏹️ {data.get('message', 'Đã dừng theo dõi')}")

            elif data.get("action") == "status_check":
                is_running = data.get("is_running", False)
                status_emoji = "✅" if is_running else "⏸️"
                formatted_parts.append(
                    f"{status_emoji} **Trạng thái hệ thống:** {'Đang chạy' if is_running else 'Đã dừng'}")

                if "health_status" in data:
                    health = data["health_status"]
                    formatted_parts.append(f"🏥 **Sức khỏe:** {health}")

        elif agent_type == "retrieval_agent":
            if data.get("action") == "data_analysis":
                query_result = data.get("query_result", {})
                formatted_parts.append(f"📊 **Kết quả truy vấn:** {query_result.get('results_count', 0)} bản ghi")
                formatted_parts.append(f"⏱️ **Thời gian xử lý:** {query_result.get('query_time', 'N/A')}")

                if "analysis" in data:
                    analysis = data["analysis"]

                    if analysis.get("summary"):
                        formatted_parts.append(f"\n📝 **Tóm tắt:** {analysis['summary']}")

                    if analysis.get("statistics"):
                        formatted_parts.append("\n📈 **Thống kê:**")
                        stats = analysis["statistics"]
                        for key, value in stats.items():
                            formatted_parts.append(f"  - {key}: {value}")

                    if analysis.get("trends"):
                        formatted_parts.append("\n📉 **Xu hướng:**")
                        for trend in analysis["trends"]:
                            formatted_parts.append(f"  - {trend}")

                    if analysis.get("patterns"):
                        formatted_parts.append("\n🔍 **Mẫu thói quen:**")
                        for pattern in analysis["patterns"]:
                            formatted_parts.append(f"  - {pattern}")

        return "\n".join(formatted_parts) if formatted_parts else json.dumps(data, ensure_ascii=False, indent=2)

    def _format_final_response(self, response: str) -> str:
        """Chuẩn hóa format phản hồi cuối cùng"""
        response = response.strip()
        response = re.sub(r'\.([A-Z])', r'. \1', response)
        response = re.sub(r'(\*\*[^*]+\*\*)', r'\n\1\n', response)
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = re.sub(r'\n\s*-\s*', '\n  - ', response)
        return response.strip()

    def _fallback_response(self, query: str, agent_responses: Dict) -> str:
        """Phản hồi dự phòng khi LLM không hoạt động"""
        response_parts = ["🤖 **Kết quả xử lý:**\n"]

        for agent_type, response in agent_responses.items():
            if response.success:
                formatted_data = self._format_agent_data(agent_type, response.data)
                response_parts.append(f"✅ **{agent_type}:**\n{formatted_data}\n")
            else:
                response_parts.append(f"❌ **{agent_type}:** Không thể xử lý - {response.error}\n")

        return "\n".join(response_parts)


class MasterAgent:
    """Master Agent chính điều khiển các sub-agent với khả năng chat thông minh"""

    def __init__(self, model_path: str = "models/Llama-3.2-3B-Instruct-Q8_0.gguf"):
        """
        Khởi tạo Master Agent
        Args:
            model_path: Đường dẫn đến mô hình LLM
        """
        self.intent_classifier = SmartIntentClassifier()
        self.llm_generator = LLMResponseGenerator(model_path)
        self.recommend_agent = None
        self.retrieval_agent = None
        self.conversation_history: List[BaseMessage] = []
        logger.info("Master Agent với khả năng chat thông minh đã được khởi tạo")

    def set_agents(self,
                   recommend_agent: Optional[Union[object, type]] = None,
                   retrieval_agent: Optional[Union[object, type]] = None):
        """
        Thiết lập các sub-agent.
        - Nếu truyền vào instance, dùng luôn instance đó.
        - Nếu truyền vào class (Type), sẽ khởi tạo mới.
        - Nếu không truyền gì, tự import và khởi tạo default.
        """
        if recommend_agent is None:
            try:
                self.recommend_agent = RecommendAgent(HealthDataCollector())
                logger.info("RecommendAgent mặc định đã được khởi tạo.")
            except Exception as e:
                logger.error(f"Không thể khởi tạo RecommendAgent: {e}")
                self.recommend_agent = None
        else:
            if isinstance(recommend_agent, type):
                try:
                    self.recommend_agent = recommend_agent()
                    logger.info("RecommendAgent custom class đã được khởi tạo.")
                except Exception as e:
                    logger.error(f"Không thể khởi tạo RecommendAgent custom: {e}")
                    self.recommend_agent = None
            else:
                self.recommend_agent = recommend_agent
                logger.info("RecommendAgent custom instance đã được gán.")

        if retrieval_agent is None:
            try:
                self.retrieval_agent = RetrievalAgent()
                logger.info("RetrievalAgent mặc định đã được khởi tạo.")
            except Exception as e:
                logger.error(f"Không thể khởi tạo RetrievalAgent: {e}")
                self.retrieval_agent = None
        else:
            if isinstance(retrieval_agent, type):
                try:
                    self.retrieval_agent = retrieval_agent()
                    logger.info("RetrievalAgent custom class đã được khởi tạo.")
                except Exception as e:
                    logger.error(f"Không thể khởi tạo RetrievalAgent custom: {e}")
                    self.retrieval_agent = None
            else:
                self.retrieval_agent = retrieval_agent
                logger.info("RetrievalAgent custom instance đã được gán.")

    async def process_query(self, query: str) -> str:
        """Xử lý câu hỏi từ người dùng với khả năng routing thông minh"""
        try:
            user_query = UserQuery(
                text=query,
                timestamp=datetime.now()
            )
            intent_result = self.intent_classifier.classify_intent(query)
            user_query.intent = intent_result["reasoning"]
            logger.info(f"Phân loại ý định: {intent_result}")

            if intent_result["use_general_chat"]:
                logger.info("Sử dụng chat chung, không cần sub-agent")
                final_response = self.llm_generator.generate_response(
                    query=query,
                    agent_responses=None,
                    context=self._get_conversation_context(),
                    use_general_chat=True
                )
                self.conversation_history.append(HumanMessage(content=query))
                self.conversation_history.append(AIMessage(content=final_response))
                return final_response

            status_message = self._create_status_message(intent_result)
            agent_responses = await self._call_agents(user_query, intent_result)

            if not any(response.success for response in agent_responses.values()):
                error_message = self._create_error_message(agent_responses)
                return f"{status_message}\n{error_message}"

            final_response = self.llm_generator.generate_response(
                query=query,
                agent_responses=agent_responses,
                context=self._get_conversation_context(),
                use_general_chat=False
            )

            self.conversation_history.append(HumanMessage(content=query))
            self.conversation_history.append(AIMessage(content=final_response))

            formatted_result = f"{status_message}\n---\n\n{final_response}"
            return formatted_result

        except Exception as e:
            logger.error(f"Lỗi xử lý câu hỏi: {e}")
            return f"❌ **Lỗi hệ thống:** Đã xảy ra lỗi khi xử lý câu hỏi - {str(e)}\n\nVui lòng thử lại sau."

    async def _call_agents(self, user_query: UserQuery, intent_result: Dict) -> Dict[str, AgentResponse]:
        """Gọi các sub-agent dựa trên ý định"""
        responses = {}
        if intent_result["primary_agent"] == AgentType.RECOMMEND or intent_result["use_both"]:
            responses["recommend_agent"] = await self._call_recommend_agent(user_query)
        if (intent_result["primary_agent"] == AgentType.RETRIEVAL or
                intent_result["use_both"] or
                intent_result["secondary_agent"] == AgentType.RETRIEVAL):
            responses["retrieval_agent"] = await self._call_retrieval_agent(user_query)
        return responses

    async def _call_recommend_agent(self, user_query: UserQuery) -> AgentResponse:
        """Gọi recommend_agent"""
        try:
            if not self.recommend_agent:
                return AgentResponse(
                    success=False,
                    error="Recommend agent chưa được khởi tạo",
                    agent_type=AgentType.RECOMMEND
                )

            query_lower = user_query.text.lower()

            if any(keyword in query_lower for keyword in ["start", "bắt đầu", "khởi động"]):
                if hasattr(self.recommend_agent, 'start'):
                    self.recommend_agent.start()
                    data = {"action": "started", "message": "Đã bắt đầu theo dõi sức khỏe"}
                else:
                    data = {"action": "not_supported", "message": "Chức năng start không được hỗ trợ"}
            elif any(keyword in query_lower for keyword in ["stop", "dừng", "tắt"]):
                if hasattr(self.recommend_agent, 'stop'):
                    self.recommend_agent.stop()
                    data = {"action": "stopped", "message": "Đã dừng theo dõi sức khỏe"}
                else:
                    data = {"action": "not_supported", "message": "Chức năng stop không được hỗ trợ"}
            elif any(keyword in query_lower for keyword in ["status", "trạng thái"]):
                status_data = {}
                if hasattr(self.recommend_agent, 'is_engine_running'):
                    status_data["is_running"] = self.recommend_agent.is_engine_running()
                if hasattr(self.recommend_agent, 'get_health_status'):
                    status_data["health_status"] = self.recommend_agent.get_health_status()
                data = {"action": "status_check", **status_data}
            else:
                if hasattr(self.recommend_agent, 'get_recommendations'):
                    data = self.recommend_agent.get_recommendations(include_analysis=True)
                    data["action"] = "recommendations"
                else:
                    data = {"action": "not_supported", "message": "Chức năng get_recommendations không được hỗ trợ"}

            return AgentResponse(
                success=True,
                data=data,
                agent_type=AgentType.RECOMMEND
            )

        except Exception as e:
            logger.error(f"Lỗi gọi recommend_agent: {e}")
            return AgentResponse(
                success=False,
                error=f"Không thể kết nối với recommend_agent: {str(e)}",
                agent_type=AgentType.RECOMMEND
            )

    async def _call_retrieval_agent(self, user_query: UserQuery) -> AgentResponse:
        """Gọi retrieval_agent"""
        try:
            if not self.retrieval_agent:
                return AgentResponse(
                    success=False,
                    error="Retrieval agent chưa được khởi tạo",
                    agent_type=AgentType.RETRIEVAL
                )

            if not hasattr(self.retrieval_agent, 'retrieve_data'):
                return AgentResponse(
                    success=False,
                    error="Retrieval agent không có method retrieve_data",
                    agent_type=AgentType.RETRIEVAL
                )

            query_result = self.retrieval_agent.retrieve_data(
                query=user_query.text,
                max_results=10
            )

            if not query_result.results:
                return AgentResponse(
                    success=False,
                    error="Không tìm thấy dữ liệu phù hợp với câu hỏi",
                    agent_type=AgentType.RETRIEVAL
                )

            analysis_result = None
            if hasattr(self.retrieval_agent, 'analyze_statistical_patterns'):
                analysis_result = self.retrieval_agent.analyze_statistical_patterns(
                    query=user_query.text,
                    data=query_result.results
                )

            data = {
                "action": "data_analysis",
                "query_result": {
                    "total_results": query_result.results,
                    "query_time": query_result.query_time,
                    "results_count": len(query_result.results)
                }
            }

            if analysis_result:
                data["analysis"] = {
                    "summary": analysis_result.summary,
                    "statistics": analysis_result.statistics,
                    "trends": analysis_result.trends,
                    "patterns": analysis_result.patterns
                }

            return AgentResponse(
                success=True,
                data=data,
                agent_type=AgentType.RETRIEVAL
            )

        except Exception as e:
            logger.error(f"Lỗi gọi retrieval_agent: {e}")
            return AgentResponse(
                success=False,
                error=f"Không thể phân tích dữ liệu: {str(e)}",
                agent_type=AgentType.RETRIEVAL
            )

    def _create_status_message(self, intent_result: Dict) -> str:
        """Tạo thông báo về việc sử dụng agent"""
        if intent_result["use_both"]:
            return "🤖 **Đang xử lý:**\n  - Recommend Agent (cảnh báo real-time)\n  - Retrieval Agent (phân tích dữ liệu)\n"
        elif intent_result["primary_agent"] == AgentType.RECOMMEND:
            return "🤖 **Đang sử dụng Recommend Agent** để lấy cảnh báo và khuyến nghị...\n"
        elif intent_result["primary_agent"] == AgentType.RETRIEVAL:
            return "🤖 **Đang sử dụng Retrieval Agent** để phân tích dữ liệu và xu hướng...\n"
        else:
            return "🤖 **Đang xử lý câu hỏi...**\n"

    def _create_error_message(self, agent_responses: Dict) -> str:
        """Tạo thông báo lỗi"""
        error_parts = ["❌ **Không thể xử lý câu hỏi do các lỗi sau:**\n"]
        for agent_type, response in agent_responses.items():
            if not response.success:
                error_parts.append(f"  - **{agent_type}:** {response.error}")
        if len(error_parts) > 1:
            return "\n".join(error_parts)
        else:
            return "❌ **Không thể xử lý câu hỏi.** Vui lòng thử lại sau.\n"

    def _get_conversation_context(self) -> str:
        """Lấy ngữ cảnh hội thoại"""
        if len(self.conversation_history) > 4:
            recent_history = self.conversation_history[-4:]
        else:
            recent_history = self.conversation_history

        context_parts = []
        for message in recent_history:
            if isinstance(message, HumanMessage):
                context_parts.append(f"Người dùng: {message.content}")
            elif isinstance(message, AIMessage):
                context_parts.append(f"Assistant: {message.content}")
        return "\n".join(context_parts)

    def reset_conversation(self):
        """Đặt lại lịch sử hội thoại"""
        self.conversation_history = []
        logger.info("Đã đặt lại lịch sử hội thoại")

    def get_agent_status(self) -> Dict:
        """Lấy trạng thái của các agent"""
        status = {
            "master_agent": "active",
            "recommend_agent": "not_connected",
            "retrieval_agent": "not_connected"
        }

        if self.recommend_agent:
            try:
                if hasattr(self.recommend_agent, 'is_engine_running'):
                    status[
                        "recommend_agent"] = "connected" if self.recommend_agent.is_engine_running() else "connected_but_stopped"
                else:
                    status["recommend_agent"] = "connected"
            except Exception as e:
                status["recommend_agent"] = f"error: {str(e)}"

        if self.retrieval_agent:
            try:
                if hasattr(self.retrieval_agent, 'get_health_status'):
                    health_status = self.retrieval_agent.get_health_status()
                    status["retrieval_agent"] = "connected" if health_status.get(
                        "status") == "healthy" else "connected_with_issues"
                else:
                    status["retrieval_agent"] = "connected"
            except Exception as e:
                status["retrieval_agent"] = f"error: {str(e)}"
        return status