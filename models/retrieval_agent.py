import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from langchain.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import statistics
from collections import Counter, defaultdict

# Import your embedding module
from models.embedding import HealthDataEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    summary: str
    trends: Dict[str, Any]
    patterns: Dict[str, Any]
    statistics: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


@dataclass
class QueryResult:
    """Data class for query results"""
    results: List[Dict[str, Any]]
    total_found: int
    query_time: float
    status: str
    analysis: Optional[AnalysisResult] = None


class RetrievalAgent:
    """
    Health Data Retrieval Agent - Tự động tạo lại vector database mỗi khi khởi tạo
    """

    def __init__(self,
                 csv_data_path: str = None,
                 model_path: str = None,
                 vector_db_dir: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 force_rebuild: bool = True):
        """
        Initialize Health Data Retrieval Agent

        Args:
            csv_data_path: Đường dẫn tới file CSV dữ liệu
            model_path: Đường dẫn tới GGUF model file
            vector_db_dir: Thư mục lưu vector database
            embedding_model: Tên model embedding
            force_rebuild: Có bắt buộc tạo lại vector DB không
        """
        # Xác định thư mục gốc project
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Đường dẫn tuyệt đối cho data/summary.csv
        if csv_data_path is None:
            self.csv_data_path = os.path.join(self.project_root, "data", "summary.csv")
        else:
            self.csv_data_path = os.path.join(self.project_root, csv_data_path) if not os.path.isabs(csv_data_path) else csv_data_path
        # Đường dẫn tuyệt đối cho vector_db_dir
        if vector_db_dir is None:
            self.vector_db_dir = os.path.join(self.project_root, "data")
        else:
            self.vector_db_dir = os.path.join(self.project_root, vector_db_dir) if not os.path.isabs(vector_db_dir) else vector_db_dir
        self.force_rebuild = force_rebuild
        # Tạo thư mục vector_db nếu chưa tồn tại
        os.makedirs(self.vector_db_dir, exist_ok=True)
        # Đường dẫn các file vector database
        self.embeddings_index_path = os.path.join(self.vector_db_dir, "health_data_index.faiss")
        self.embeddings_metadata_path = os.path.join(self.vector_db_dir, "health_data_metadata.pkl")
        self.embeddings_scaler_path = os.path.join(self.vector_db_dir, "health_data_scaler.pkl")
        # Đường dẫn tuyệt đối cho model_path
        if model_path is None:
            self.model_path = os.path.join(self.project_root,  "models/Llama-3.2-3B-Instruct-Q8_0.gguf")
        else:
            self.model_path = os.path.join(self.project_root, model_path) if not os.path.isabs(model_path) else model_path

        # Initialize embedding system
        self.embedder = HealthDataEmbedding(model_name=embedding_model)

        # Tạo hoặc load vector database
        self._setup_vector_database()

        # Initialize LLM
        self.llm = self._init_llm(self.model_path)

        # Initialize analysis chains
        self.statistical_analysis_chain = self._init_statistical_analysis_chain()
        self.trend_analysis_chain = self._init_trend_analysis_chain()
        self.pattern_analysis_chain = self._init_pattern_analysis_chain()

        logger.info("Health Data Retrieval Agent initialized successfully")

    def _setup_vector_database(self):
        """
        Tạo lại vector database từ CSV hoặc load từ file đã lưu
        """
        should_rebuild = self.force_rebuild or not self._vector_db_exists()

        if should_rebuild:
            logger.info("Tạo lại vector database từ CSV data...")
            self._create_vector_database()
        else:
            logger.info("Load vector database đã có...")
            self._load_vector_database()

    def _vector_db_exists(self) -> bool:
        """Kiểm tra xem vector database đã tồn tại chưa"""
        return (os.path.exists(self.embeddings_index_path) and
                os.path.exists(self.embeddings_metadata_path) and
                os.path.exists(self.embeddings_scaler_path))

    def _create_vector_database(self):
        """Tạo vector database từ CSV data"""
        try:
            # Kiểm tra file CSV có tồn tại không
            if not os.path.exists(self.csv_data_path):
                raise FileNotFoundError(f"Không tìm thấy file CSV: {self.csv_data_path}")

            # Load dữ liệu từ CSV
            logger.info(f"Loading data from {self.csv_data_path}")
            df = self.embedder.load_data(self.csv_data_path)

            if df.empty:
                raise ValueError("CSV file trống hoặc không đọc được")

            # Tạo embeddings
            logger.info("Creating embeddings...")
            self.embedder.create_embeddings(df)

            # Lưu vector database
            logger.info("Saving vector database...")
            self.embedder.save_index(
                index_path=self.embeddings_index_path,
                metadata_path=self.embeddings_metadata_path,
                scaler_path=self.embeddings_scaler_path
            )

            logger.info(f"Vector database created successfully with {len(self.embedder.metadata)} sessions")

        except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            raise

    def _load_vector_database(self):
        """Load vector database từ file đã lưu"""
        try:
            self.embedder.load_index(
                index_path=self.embeddings_index_path,
                metadata_path=self.embeddings_metadata_path,
                scaler_path=self.embeddings_scaler_path
            )
            logger.info(f"Loaded vector database with {len(self.embedder.metadata)} sessions")
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            logger.info("Falling back to creating new database...")
            self._create_vector_database()

    def _init_llm(self, model_path: str):
        """Initialize LLM with appropriate settings"""
        try:
            # Kiểm tra file model có tồn tại không
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return None

            # Optimized settings for RTX 3050 (4GB VRAM)
            llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=-1,
                n_batch=512,
                n_ctx=2048,
                temperature=0.5,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                max_tokens=1024,
                verbose=False,
                n_threads=4,
                use_mmap=True,
                use_mlock=True
            )
            logger.info("LLM initialized successfully")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Fallback to CPU-only mode
            try:
                return LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=0,
                    n_batch=256,
                    n_ctx=1024,
                    temperature=0.3,
                    max_tokens=512,
                    verbose=False,
                    n_threads=6
                )
            except:
                logger.warning("LLM initialization failed completely. Analysis will be limited.")
                return None

    def _init_statistical_analysis_chain(self):
        if self.llm is None:
            raise RuntimeError("LLM model not initialized. Cannot create LLMChain. Kiểm tra lại file model hoặc cấu hình máy.")
        prompt = PromptTemplate(
            input_variables=["query", "statistics", "context"],
            template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id>
    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024
    Bạn là một chuyên gia phân tích dữ liệu thống kê sức khỏe khi làm việc với máy tính.
    Dựa vào câu hỏi và dữ liệu thống kê được cung cấp, hãy đưa ra phân tích chi tiết.<|eot_id|><|start_header_id|>user<|end_header_id>
    Câu hỏi: {query}
    Dữ liệu thống kê: {statistics}
    Ngữ cảnh: {context}
    Hãy phân tích và đưa ra:
    1. Tổng quan về dữ liệu
    2. Các chỉ số quan trọng
    3. So sánh với các chuẩn mực
    4. Đánh giá tổng thể
    Trả lời bằng tiếng Việt, tập trung vào các số liệu và phân tích thống kê:<|eot_id|><|start_header_id|>assistant<|end_header_id>
    """
        )
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )
    def _init_trend_analysis_chain(self):
        if self.llm is None:
            raise RuntimeError("LLM model not initialized. Cannot create LLMChain. Kiểm tra lại file model hoặc cấu hình máy.")
        prompt = PromptTemplate(
            input_variables=["query", "trends", "time_series"],
            template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id>
    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024
    Bạn là một chuyên gia phân tích xu hướng sức khỏe khi làm việc với máy tính.
    Dựa vào câu hỏi và dữ liệu xu hướng được cung cấp, hãy đưa ra phân tích chi tiết.<|eot_id|><|start_header_id|>user<|end_header_id>
    Câu hỏi: {query}
    Dữ liệu xu hướng: {trends}
    Chuỗi thời gian: {time_series}
    Hãy phân tích và đưa ra:
    1. Xu hướng chính
    2. Các yếu tố ảnh hưởng
    3. Đề xuất cải thiện
    Trả lời bằng tiếng Việt, tập trung vào các xu hướng:<|eot_id|><|start_header_id|>assistant<|end_header_id>
    """
        )
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )
    def _init_pattern_analysis_chain(self):
        if self.llm is None:
            raise RuntimeError("LLM model not initialized. Cannot create LLMChain. Kiểm tra lại file model hoặc cấu hình máy.")
        prompt = PromptTemplate(
            input_variables=["query", "patterns", "behaviors"],
            template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id>
    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024
    Bạn là một chuyên gia phân tích hành vi và thói quen sức khỏe khi làm việc với máy tính.
    Dựa vào câu hỏi và dữ liệu hành vi được cung cấp, hãy đưa ra phân tích chi tiết.<|eot_id|><|start_header_id|>user<|end_header_id>
    Câu hỏi: {query}
    Dữ liệu hành vi: {patterns}
    Thói quen: {behaviors}
    Hãy phân tích và đưa ra:
    1. Các mẫu hành vi chính
    2. Thói quen tích cực và tiêu cực
    3. Mối quan hệ giữa các yếu tố
    4. Độ ổn định của hành vi
    Trả lời bằng tiếng Việt, tập trung vào các mẫu hành vi và thói quen:<|eot_id|><|start_header_id|>assistant<|end_header_id>
    """
        )
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )

    def retrieve_data(self, query: str, max_results: int = 5, filters: Optional[Dict] = None) -> QueryResult:
        """
        Retrieve health data based on query

        Args:
            query: Search query
            max_results: Maximum number of results (default is 5)
            filters: Additional filters

        Returns:
            QueryResult object with retrieved data
        """
        start_time = datetime.now()
        try:
            # Text-based search
            text_results = self.embedder.search_similar_sessions(query, k=max_results)

            # Metric-based search if filters provided
            metric_results = []
            if filters:
                metric_results = self.embedder.search_by_metrics(
                    posture_score_min=filters.get('posture_score_min'),
                    attention_score_min=filters.get('attention_score_min'),
                    duration_min=filters.get('duration_min'),
                    hour_range=filters.get('hour_range')
                )

            # Combine results, prioritizing text search, limit to 5
            all_results = []
            seen_ids = set()

            # Add text results first
            for result in text_results:
                session_id = result['metadata']['session_id']
                if session_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(session_id)
                    if len(all_results) >= 5:
                        break

            # If less than 5, add from metric results
            if len(all_results) < 5 and metric_results:
                for meta in metric_results:
                    session_id = meta['session_id']
                    if session_id not in seen_ids:
                        all_results.append({
                            'rank': len(all_results) + 1,
                            'similarity_score': 0.0,
                            'metadata': meta
                        })
                        seen_ids.add(session_id)
                        if len(all_results) >= 5:
                            break

            query_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                results=all_results,
                total_found=len(all_results),
                query_time=query_time,
                status='success' if all_results else 'no_results'
            )
        except Exception as e:
            logger.error(f"Data retrieval error: {e}")
            return QueryResult(
                results=[],
                total_found=0,
                query_time=(datetime.now() - start_time).total_seconds(),
                status='error'
            )

    def summarize_stats(self, stats: Dict) -> str:
        if not stats:
            return "Không có dữ liệu thống kê"
        summary = []
        for metric, values in stats.items():
            if metric != 'overall':
                summary.append(
                    f"{metric}: trung bình {values['mean']:.2f}, min {values['min']:.2f}, max {values['max']:.2f}")
        overall = stats.get('overall', {})
        summary.append(f"Tổng số phiên: {overall.get('total_sessions', 0)}")
        summary.append(f"Thời gian làm việc trung bình: {overall.get('avg_session_duration', 0):.2f} phút")
        return "\n".join(summary)

    def summarize_trends(self, trends: Dict) -> str:
        if not trends:
            return "Không có dữ liệu xu hướng"
        summary = []
        if 'daily' in trends:
            daily = trends['daily']
            if daily:
                summary.append("Xu hướng hàng ngày:")
                for day, data in list(daily.items())[:3]:
                    summary.append(f"- {day}: posture {data['avg_posture']:.2f}, attention {data['avg_attention']:.2f}")
        if 'weekly' in trends:
            weekly = trends['weekly']
            if weekly:
                summary.append("Xu hướng hàng tuần:")
                for week, data in list(weekly.items())[:2]:
                    summary.append(
                        f"- {week}: posture {data['avg_posture']:.2f}, attention {data['avg_attention']:.2f}")
        return "\n".join(summary)

    def summarize_patterns(self, patterns: Dict) -> str:
        if not patterns:
            return "Không có dữ liệu hành vi"
        summary = []
        if 'preferred_hours' in patterns:
            hours = patterns['preferred_hours']
            top_hours = sorted(hours.items(), key=lambda x: x[1], reverse=True)[:3]
            summary.append("Giờ làm việc phổ biến: " + ", ".join(f"{h}: {c}" for h, c in top_hours))
        if 'performance_by_hour' in patterns:
            perf = patterns['performance_by_hour']
            if perf:
                best_hour = max(perf.items(), key=lambda x: x[1]['avg_attention'])
                summary.append(
                    f"Giờ tập trung tốt nhất: {best_hour[0]}h với attention {best_hour[1]['avg_attention']:.2f}")
        return "\n".join(summary)

    def analyze_statistical_patterns(self, query: str, data: List[Dict]) -> AnalysisResult:
        """Analyze statistical patterns in the data"""
        try:
            logger.info("Starting statistical analysis")
            stats = self._calculate_comprehensive_statistics(data)
            logger.info("Calculated statistics")
            trends = self._analyze_trends(data)
            logger.info("Analyzed trends")
            patterns = self._identify_patterns(data)

            # Phân tích thống kê với dữ liệu tóm tắt
            stat_analysis = self.statistical_analysis_chain.run(
                query=query,
                statistics=self.summarize_stats(stats),
                context=self._prepare_context(data)
            )

            # Phân tích xu hướng với dữ liệu tóm tắt
            trend_analysis = self.trend_analysis_chain.run(
                query=query,
                trends=self.summarize_trends(trends),
                time_series=self._prepare_time_series(data)
            )

            # Phân tích hành vi với dữ liệu tóm tắt
            pattern_analysis = self.pattern_analysis_chain.run(
                query=query,
                patterns=self.summarize_patterns(patterns),
                behaviors=self._prepare_behaviors(data)
            )

            # Kết hợp kết quả với định dạng dễ đọc hơn
            combined_summary = f"""
    **PHÂN TÍCH THỐNG KÊ**

    {stat_analysis}


    **PHÂN TÍCH XU HƯỚNG**

    {trend_analysis}


    **PHÂN TÍCH HÀNH VI**

    {pattern_analysis}
    """

            return AnalysisResult(
                summary=combined_summary.strip(),
                trends=trends,
                patterns=patterns,
                statistics=stats,
                recommendations=[],
                confidence_score=self._calculate_confidence_score(data)
            )

        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return AnalysisResult(
                summary=f"Lỗi khi phân tích thống kê: {str(e)}",
                trends={},
                patterns={},
                statistics={},
                recommendations=[],
                confidence_score=0.0
            )
    def _calculate_comprehensive_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from data"""
        if not data:
            logger.warning("No data provided for statistical calculation")
            return {}

        stats = {}
        # Khởi tạo các danh sách để trích xuất dữ liệu
        posture_scores = []
        attention_scores = []
        durations = []
        blink_rates = []
        distances = []
        bad_posture_counts = []
        drowsiness_counts = []
        timestamps = []

        # Trích xuất dữ liệu với kiểm tra an toàn
        for result in data:
            meta = result.get('metadata', {})
            health = meta.get('health_metrics', {})
            derived = meta.get('derived_metrics', {})

            # Chỉ thêm giá trị nếu trường tồn tại và là số hợp lệ
            try:
                if 'posture_quality_score' in derived:
                    posture_scores.append(float(derived['posture_quality_score']))
                if 'attention_score' in derived:
                    attention_scores.append(float(derived['attention_score']))
                if 'session_duration_minutes' in meta:
                    durations.append(float(meta['session_duration_minutes']))
                if 'blink_per_minute' in health:
                    blink_rates.append(float(health['blink_per_minute']))
                if 'avg_distance' in health:
                    distances.append(float(health['avg_distance']))
                if 'bad_posture_count' in health:
                    bad_posture_counts.append(float(health['bad_posture_count']))
                if 'drowsiness_count' in health:
                    drowsiness_counts.append(float(health['drowsiness_count']))
                if 'timestamp' in meta:
                    timestamps.append(meta['timestamp'])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid data type in metadata: {e}")
                continue

        # Tính toán thống kê với kiểm tra danh sách rỗng
        metrics = {
            'posture_scores': posture_scores,
            'attention_scores': attention_scores,
            'durations': durations,
            'blink_rates': blink_rates,
            'distances': distances,
            'bad_posture_counts': bad_posture_counts,
            'drowsiness_counts': drowsiness_counts
        }

        for metric_name, values in metrics.items():
            if values:  # Chỉ tính nếu danh sách không rỗng
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'count': len(values)
                }
            else:
                stats[metric_name] = {
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0,
                    'q25': 0, 'q75': 0, 'count': 0
                }

        # Thống kê tổng quan
        stats['overall'] = {
            'total_sessions': len(data),
            'total_duration_hours': sum(durations) / 60 if durations else 0,
            'avg_session_duration': float(np.mean(durations)) if durations else 0,
            'health_score_avg': (float(np.mean(posture_scores)) + float(
                np.mean(attention_scores))) / 2 if posture_scores and attention_scores else 0,
            'time_range': {
                'start': min(timestamps) if timestamps else None,
                'end': max(timestamps) if timestamps else None
            }
        }

        return stats

    def _analyze_trends(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        if not data:
            logger.warning("No data provided for trend analysis")
            return {}

        trends = {}
        try:
            # Trích xuất dữ liệu với timestamp
            timestamped_data = []
            for result in data:
                meta = result.get('metadata', {})
                timestamp = self._get_timestamp_from_metadata(meta)
                if timestamp:
                    derived = meta.get('derived_metrics', {})
                    timestamped_data.append({
                        'timestamp': timestamp,
                        'posture': derived.get('posture_quality_score', 0),
                        'attention': derived.get('attention_score', 0),
                        'duration': meta.get('session_duration_minutes', 0)
                    })

            if not timestamped_data:
                logger.warning("No valid timestamp data found for trend analysis")
                return {'error': 'No valid timestamp data found'}

            # Sắp xếp theo timestamp
            timestamped_data.sort(key=lambda x: x['timestamp'])

            # Tính toán xu hướng theo ngày, tuần, giờ
            daily_stats = defaultdict(list)
            weekly_stats = defaultdict(list)
            hourly_stats = defaultdict(list)

            for item in timestamped_data:
                timestamp = item['timestamp']
                day_key = timestamp.strftime('%Y-%m-%d')
                week_key = timestamp.strftime('%Y-W%U')
                hour_key = timestamp.hour

                session_data = {
                    'posture': item['posture'],
                    'attention': item['attention'],
                    'duration': item['duration']
                }

                daily_stats[day_key].append(session_data)
                weekly_stats[week_key].append(session_data)
                hourly_stats[hour_key].append(session_data)

            # Tính xu hướng hàng ngày
            trends['daily'] = {}
            for day, sessions in daily_stats.items():
                posture_scores = [s['posture'] for s in sessions if s['posture'] > 0]
                attention_scores = [s['attention'] for s in sessions if s['attention'] > 0]
                durations = [s['duration'] for s in sessions if s['duration'] > 0]
                trends['daily'][day] = {
                    'avg_posture': float(np.mean(posture_scores)) if posture_scores else 0,
                    'avg_attention': float(np.mean(attention_scores)) if attention_scores else 0,
                    'total_duration': sum(durations),
                    'session_count': len(sessions)
                }

            # Tính xu hướng hàng tuần
            trends['weekly'] = {}
            for week, sessions in weekly_stats.items():
                posture_scores = [s['posture'] for s in sessions if s['posture'] > 0]
                attention_scores = [s['attention'] for s in sessions if s['attention'] > 0]
                durations = [s['duration'] for s in sessions if s['duration'] > 0]
                trends['weekly'][week] = {
                    'avg_posture': float(np.mean(posture_scores)) if posture_scores else 0,
                    'avg_attention': float(np.mean(attention_scores)) if attention_scores else 0,
                    'total_duration': sum(durations),
                    'session_count': len(sessions)
                }

            # Tính xu hướng theo giờ
            trends['hourly'] = {}
            for hour, sessions in hourly_stats.items():
                posture_scores = [s['posture'] for s in sessions if s['posture'] > 0]
                attention_scores = [s['attention'] for s in sessions if s['attention'] > 0]
                durations = [s['duration'] for s in sessions if s['duration'] > 0]
                trends['hourly'][hour] = {
                    'avg_posture': float(np.mean(posture_scores)) if posture_scores else 0,
                    'avg_attention': float(np.mean(attention_scores)) if attention_scores else 0,
                    'avg_duration': float(np.mean(durations)) if durations else 0,
                    'session_count': len(sessions)
                }

            return trends

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'error': str(e)}

    def _identify_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Identify behavioral patterns in the data"""
        if not data:
            logger.warning("No data provided for pattern identification")
            return {}

        patterns = {}
        try:
            # Trích xuất thời gian làm việc
            work_hours = []
            work_days = []
            valid_timestamps = []

            for result in data:
                meta = result.get('metadata', {})
                timestamp = self._get_timestamp_from_metadata(meta)
                if timestamp:
                    work_hours.append(timestamp.hour)
                    work_days.append(timestamp.weekday())
                    valid_timestamps.append(timestamp)
                else:
                    time_features = meta.get('time_features', {})
                    if 'hour_of_day' in time_features:
                        work_hours.append(int(time_features['hour_of_day']))

            if not work_hours:
                logger.warning("No valid time data found for pattern identification")
                return {'error': 'No valid time data found'}

            # Giờ làm việc phổ biến
            hour_counter = Counter(work_hours)
            patterns['preferred_hours'] = dict(hour_counter.most_common(5))

            # Ngày làm việc phổ biến
            if work_days:
                day_counter = Counter(work_days)
                day_names = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
                patterns['preferred_days'] = {
                    day_names[day]: count for day, count in day_counter.most_common()
                    if 0 <= day < len(day_names)
                }

            # Hiệu suất theo giờ
            posture_by_hour = defaultdict(list)
            attention_by_hour = defaultdict(list)

            for result in data:
                meta = result.get('metadata', {})
                timestamp = self._get_timestamp_from_metadata(meta)
                if timestamp:
                    hour = timestamp.hour
                    derived = meta.get('derived_metrics', {})
                    if 'posture_quality_score' in derived:
                        posture_by_hour[hour].append(float(derived['posture_quality_score']))
                    if 'attention_score' in derived:
                        attention_by_hour[hour].append(float(derived['attention_score']))

            patterns['performance_by_hour'] = {}
            for hour in range(24):
                if hour in posture_by_hour or hour in attention_by_hour:
                    patterns['performance_by_hour'][hour] = {
                        'avg_posture': float(np.mean(posture_by_hour[hour])) if posture_by_hour[hour] else 0,
                        'avg_attention': float(np.mean(attention_by_hour[hour])) if attention_by_hour[hour] else 0,
                        'session_count': max(len(posture_by_hour[hour]), len(attention_by_hour[hour]))
                    }

            # Độ dài phiên
            durations = []
            for result in data:
                meta = result.get('metadata', {})
                duration = meta.get('session_duration_minutes', 0)
                if duration > 0:
                    durations.append(float(duration))

            if durations:
                patterns['session_duration'] = {
                    'short_sessions': len([d for d in durations if d < 30]),
                    'medium_sessions': len([d for d in durations if 30 <= d < 90]),
                    'long_sessions': len([d for d in durations if d >= 90]),
                    'avg_duration': float(np.mean(durations))
                }

            return patterns

        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return {'error': str(e)}

    def _get_timestamp_from_metadata(self, metadata: Dict) -> Optional[datetime]:
        """Extract timestamp from metadata safely"""
        try:
            # Thử lấy timestamp từ các trường có thể có
            if 'timestamp' in metadata:
                timestamp_str = metadata['timestamp']
            elif 'begin_timestamp' in metadata:
                timestamp_str = metadata['begin_timestamp']
            elif 'end_timestamp' in metadata:
                timestamp_str = metadata['end_timestamp']
            else:
                return None

            # Chuyển đổi string thành datetime
            if isinstance(timestamp_str, str):
                # Thử các format khác nhau
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%dT%H:%M:%S.%f'
                ]

                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue

                # Nếu không parse được, thử dùng fromisoformat
                try:
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    return None

            return None
        except Exception as e:
            logger.warning(f"Error extracting timestamp: {e}")
            return None

    def _prepare_context(self, data: List[Dict]) -> str:
        """Prepare context for analysis"""
        if not data:
            return "Không có dữ liệu ngữ cảnh"

        context = f"Phân tích dựa trên {len(data)} phiên làm việc "

        # Time range - sử dụng hàm mới để lấy timestamp
        timestamps = []
        for result in data:
            timestamp = self._get_timestamp_from_metadata(result['metadata'])
            if timestamp:
                timestamps.append(timestamp)

        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
            context += f"từ {start_time.strftime('%Y-%m-%d %H:%M:%S')} đến {end_time.strftime('%Y-%m-%d %H:%M:%S')}"

        return context
    def _prepare_time_series(self, data: List[Dict]) -> str:
        """Prepare time series data for analysis"""
        if not data:
            return "Không có dữ liệu chuỗi thời gian"

        # Sort by timestamp
        valid_data = []
        for result in data:
            timestamp = self._get_timestamp_from_metadata(result['metadata'])
            if timestamp:
                valid_data.append((timestamp, result))

        if not valid_data:
            return "Không có dữ liệu timestamp hợp lệ"

        # Sort by timestamp
        sorted_data = sorted(valid_data, key=lambda x: x[0])

        time_series = []
        for timestamp, result in sorted_data:
            meta = result['metadata']
            derived = meta.get('derived_metrics', {})

            time_series.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'posture': derived.get('posture_quality_score', 0),
                'attention': derived.get('attention_score', 0),
                'duration': meta.get('session_duration_minutes', 0)
            })

        return json.dumps(time_series[:10], indent=2, ensure_ascii=False)  # Limit to 10 for context

    def _prepare_behaviors(self, data: List[Dict]) -> str:
        """Prepare behavior data for analysis"""
        if not data:
            return "Không có dữ liệu hành vi"

        behaviors = []
        for result in data:
            meta = result['metadata']
            health = meta['health_metrics']

            behaviors.append({
                'session_id': meta['session_id'],
                'blink_rate': health['blink_per_minute'],
                'avg_distance': health['avg_distance'],
                'bad_posture_count': health['bad_posture_count'],
                'drowsiness_count': health['drowsiness_count']
            })

        return json.dumps(behaviors[:5], indent=2, ensure_ascii=False)  # Limit to 5 for context

    def _calculate_confidence_score(self, data: List[Dict]) -> float:
        """Calculate confidence score for analysis"""
        if not data:
            return 0.0

        # Base confidence on data quantity and quality
        quantity_score = min(len(data) / 50, 1.0)  # Max confidence at 50+ sessions

        # Quality score based on data completeness
        complete_sessions = 0
        for result in data:
            meta = result['metadata']
            meta = result['metadata']
            if all(key in meta for key in ['health_metrics', 'derived_metrics', 'session_duration_minutes']):
                complete_sessions += 1

        quality_score = complete_sessions / len(data) if data else 0.0

        # Overall confidence
        confidence = (quantity_score * 0.6 + quality_score * 0.4)

        return round(confidence, 2)

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all data"""
        try:
            stats = self.embedder.get_statistics()

            # Add more detailed statistics
            all_metadata = self.embedder.metadata

            if all_metadata:
                # Calculate detailed metrics
                detailed_stats = self._calculate_comprehensive_statistics(
                    [{'metadata': meta} for meta in all_metadata]
                )

                stats.update(detailed_stats)

            return stats

        except Exception as e:
            logger.error(f"Error getting comprehensive statistics: {e}")
            return {}

    def search_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Search sessions within specific time range"""
        try:
            results = []

            for meta in self.embedder.metadata:
                session_time = self._get_timestamp_from_metadata(meta)

                if session_time and start_time <= session_time <= end_time:
                    results.append({
                        'rank': len(results) + 1,
                        'similarity_score': 1.0,
                        'metadata': meta
                    })

            return results

        except Exception as e:
            logger.error(f"Error searching by time range: {e}")
            return []

    def search_by_performance_metrics(self,
                                      min_posture: float = 0,
                                      max_posture: float = 100,
                                      min_attention: float = 0,
                                      max_attention: float = 100) -> List[Dict]:
        """Search sessions by performance metrics"""
        try:
            results = []

            for meta in self.embedder.metadata:
                derived = meta['derived_metrics']
                posture = derived['posture_quality_score']
                attention = derived['attention_score']

                if (min_posture <= posture <= max_posture and
                        min_attention <= attention <= max_attention):
                    results.append({
                        'rank': len(results) + 1,
                        'similarity_score': 1.0,
                        'metadata': meta
                    })

            return results

        except Exception as e:
            logger.error(f"Error searching by performance metrics: {e}")
            return []

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent"""
        try:
            return {
                'status': 'healthy',
                'embeddings_loaded': self.embedder.index is not None,
                'llm_loaded': self.llm is not None,
                'total_sessions': len(self.embedder.metadata) if self.embedder.metadata else 0,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }


