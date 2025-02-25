# CyberGirl

## Project Structure

```
cyberGirl/
├── config/                         # 配置中心
│   ├── model_config.yaml           # 模型超参数配置
│   ├── database_config.py          # 数据库连接配置
│   └── character_settings/         # 角色配置文件
│       ├── base_persona.yaml       # 基础人设
│       └── scenario_prompts/       # 场景提示词模板
├── core/                           # 核心逻辑
│   ├── generation/                 # 生成模块
│   │   ├── model_loader.py         # 模型加载器
│   │   └── response_generator.py   # 响应生成器
│   ├── memory/                     # 记忆系统
│   │   ├── memory_manager.py       # 记忆管理
│   │   └── memory_retriever.py     # 记忆检索
│   ├── rag/                        # RAG模块
│   │   ├── knowledge_loader.py     # 知识库加载
│   │   └── vector_retriever.py     # 向量检索
│   └── user_manager.py             # 用户会话管理
├── data/                           # 数据存储
│   ├── databases/                  # 结构化数据库
│   │   └── user_memories.db        # SQLite数据库文件
│   └── vector_stores/              # 向量数据库
│       └── role_knowledge.faiss    # FAISS索引
├── knowledge/                      # 知识库源文件
│   ├── character_profile.md        # 角色背景设定
│   ├── dialogue_style.md           # 对话风格指南
│   └── scenarios/                  # 场景知识
│       ├── daily_chat.md
│       └── dating_advice.md
├── server/                         # API服务
│   ├── api_server.py               # FastAPI主服务
│   └── schemas.py                  # API数据模型
├── tests/                          # 测试模块
│   ├── unit_tests/                 # 单元测试
│   └── integration_tests/          # 集成测试
├── utils/                          # 工具函数
│   ├── data_processor.py           # 数据处理工具
│   ├── logger_config.py            # 日志配置
│   └── security.py                 # 加密模块
├── requirements.txt                # Python依赖
├── Dockerfile                      # 容器化部署
├── .env                            # 环境变量
├── startup.sh                      # 启动脚本
└── .gitignore                      # 版本控制忽略
```