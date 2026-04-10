# 📚 Octi AI Trading Bot - Documentation Index

**Last Updated:** April 10, 2026  
**Phase:** 2 - Documentation Organization

---

## 🎯 Quick Navigation

- 📋 **[Getting Started](#getting-started)** - Start here if you're new
- 🏗️ **[Architecture](#architecture)** - System design and components
- 🔌 **[API Documentation](#api-documentation)** - Module interfaces
- 🚀 **[Deployment](#deployment)** - How to run the system
- 📖 **[Guides](#guides)** - How-to articles and tutorials
- 🔍 **[Reference](#reference)** - Quick reference materials

---

## Getting Started

### First Time Setup
1. Read the [**Project Overview**](./guides/01_PROJECT_OVERVIEW.md)
2. Follow the [**Installation Guide**](./guides/02_INSTALLATION.md)
3. Check the [**Quick Start**](./guides/03_QUICK_START.md)

### Understanding the Project
- **Project Structure:** See [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Agents Overview:** See [agents module documentation](../agents/__init__.py)
- **Core Components:** See [core module documentation](../core/)

---

## Architecture

### System Overview
- [System Architecture](./architecture/01_SYSTEM_ARCHITECTURE.md)
- [Module Structure](./architecture/02_MODULE_STRUCTURE.md)
- [Agent Interactions](./architecture/03_AGENT_INTERACTIONS.md)
- [Data Flow](./architecture/04_DATA_FLOW.md)

### Core Modules
```
octivault_trader/
├── agents/              # AI Trading Agents
│   ├── dip_sniper.py
│   ├── ipo_chaser.py
│   ├── liquidation_agent.py
│   ├── ml_forecaster.py
│   ├── swing_trade_hunter.py
│   ├── symbol_screener.py
│   ├── trend_hunter.py
│   └── wallet_scanner_agent.py
│
├── core/                # Core Trading Engine
│   ├── agent_manager.py
│   ├── app_context.py
│   ├── cash_router.py
│   ├── execution_manager.py
│   ├── risk_manager.py
│   └── ... (12+ core modules)
│
├── models/              # ML Models
├── config/              # Configuration
├── utils/               # Utilities
└── tests/               # Test Suite
```

---

## API Documentation

### Module APIs
- [Agents Module](./api/01_AGENTS_API.md)
- [Core Module](./api/02_CORE_API.md)
- [Models Module](./api/03_MODELS_API.md)
- [Config Module](./api/04_CONFIG_API.md)
- [Utils Module](./api/05_UTILS_API.md)

### Key Classes
- **Agent System**
  - `AgentManager` - Central agent coordination
  - `Agent` - Base agent class
  - `ActionRouter` - Action routing and execution

- **Execution System**
  - `ExecutionManager` - Trade execution
  - `PositionManager` - Position tracking
  - `RiskManager` - Risk control

- **Market Data**
  - `MarketDataFeed` - Real-time market data
  - `StreamManager` - Data streaming

---

## Deployment

### Production Deployment
- [Deployment Guide](./deployment/01_DEPLOYMENT_GUIDE.md)
- [Docker Setup](./deployment/02_DOCKER_SETUP.md)
- [Configuration](./deployment/03_CONFIGURATION.md)
- [Monitoring](./deployment/04_MONITORING.md)

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the system
python main.py
```

---

## Guides

### Development
- [Development Setup](./guides/04_DEVELOPMENT_SETUP.md)
- [Contributing Guide](./guides/05_CONTRIBUTING.md)
- [Testing Guide](./guides/06_TESTING.md)
- [Code Style Guide](./guides/07_CODE_STYLE.md)

### Operations
- [Running the Bot](./guides/10_RUNNING_THE_BOT.md)
- [Monitoring & Logging](./guides/11_MONITORING.md)
- [Troubleshooting](./guides/12_TROUBLESHOOTING.md)
- [Common Issues](./guides/13_COMMON_ISSUES.md)

### Advanced
- [Custom Agents](./guides/20_CUSTOM_AGENTS.md)
- [Strategy Development](./guides/21_STRATEGY_DEVELOPMENT.md)
- [Performance Tuning](./guides/22_PERFORMANCE_TUNING.md)
- [API Integration](./guides/23_API_INTEGRATION.md)

---

## Reference

### Quick Reference
- [Command Reference](./reference/01_COMMAND_REFERENCE.md)
- [Configuration Reference](./reference/02_CONFIG_REFERENCE.md)
- [Environment Variables](./reference/03_ENV_VARIABLES.md)

### Technical Reference
- [Algorithm Details](./reference/10_ALGORITHM_DETAILS.md)
- [Database Schema](./reference/11_DATABASE_SCHEMA.md)
- [API Endpoints](./reference/12_API_ENDPOINTS.md)

### Glossary
- [Trading Terms](./reference/20_TRADING_GLOSSARY.md)
- [System Terms](./reference/21_SYSTEM_GLOSSARY.md)

---

## 📂 Directory Structure

```
docs/
├── README.md (this file)         # Main documentation index
├── guides/                        # How-to guides and tutorials
├── architecture/                  # System design documentation
├── api/                          # API and module documentation
├── deployment/                   # Deployment and operations guides
└── reference/                    # Quick reference materials
```

---

## 🔗 Related Files

### In Root Directory
- **ARCHITECTURE.md** - High-level system architecture
- **README.md** - Project overview
- **requirements.txt** - Python dependencies
- **setup.py** - Package setup configuration

### In Project Directories
- **agents/__init__.py** - Agents module documentation
- **core/__init__.py** - Core module documentation
- **models/__init__.py** - Models module documentation
- **config/__init__.py** - Configuration module documentation

---

## 📞 Support

### Getting Help
1. Check the [Troubleshooting](./guides/12_TROUBLESHOOTING.md) guide
2. Review [Common Issues](./guides/13_COMMON_ISSUES.md)
3. Check existing GitHub issues
4. Create a new issue with detailed information

### Reporting Issues
When reporting issues, include:
- Python version
- System information
- Steps to reproduce
- Error messages and logs
- Expected vs actual behavior

---

## 📝 Documentation Standards

All documentation should follow these standards:
- **Clear headings** with proper hierarchy
- **Code examples** where applicable
- **Links** to related documents
- **Table of contents** for long documents
- **Version information** when relevant

---

## 🔄 Documentation Updates

This documentation is maintained alongside the code. When making changes:
1. Update relevant docs
2. Update this index if adding new sections
3. Include docs in pull requests
4. Keep examples up to date

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-04-10 | 2.0 | Phase 2 organization complete |
| 2026-04-10 | 1.1 | Phase 1 package structure fixes |
| 2026-03-XX | 1.0 | Initial documentation |

---

**Last Updated:** April 10, 2026  
**Status:** 📝 Complete & Organized  
**Next Review:** After Phase 3 (Repository Cleanup)
