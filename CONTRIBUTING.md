# Contributing to Budget Finetuning Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU, etc.)
- Error messages and stack traces

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach (if applicable)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**: `git commit -m "Add: feature description"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/budget-finetune-pipeline-qlora-spectrum.git
cd budget-finetune-pipeline-qlora-spectrum

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 100 characters (not 80)
- Use type hints where appropriate
- Docstrings for all public functions/classes

### Formatting

```bash
# Format code with black
black src/ *.py

# Check with flake8
flake8 src/ --max-line-length=100

# Type checking with mypy
mypy src/
```

### Documentation

- Add docstrings to all functions and classes
- Update README.md if adding new features
- Add examples for new functionality
- Comment complex logic

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_memory_tracker.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Write tests for new features
- Maintain or improve code coverage
- Use descriptive test names
- Test edge cases and error conditions

Example test structure:
```python
def test_memory_tracker_initialization():
    """Test that MemoryTracker initializes correctly"""
    tracker = MemoryTracker()
    assert tracker is not None
    assert hasattr(tracker, 'peak_memory')
```

## Areas for Contribution

### High Priority

1. **Additional Finetuning Methods**
   - DoRA (Weight-Decomposed LoRA)
   - AdaLoRA (Adaptive LoRA)
   - IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
   
2. **Model Support**
   - Mistral models
   - Falcon models
   - MPT models
   
3. **Evaluation Metrics**
   - BLEU/ROUGE scores
   - Human evaluation framework
   - Task-specific benchmarks

4. **Multi-GPU Support**
   - DeepSpeed integration
   - FSDP (Fully Sharded Data Parallel)
   - Distributed training

### Medium Priority

5. **Dataset Support**
   - More dataset loaders
   - Custom dataset templates
   - Data preprocessing utilities

6. **Visualization**
   - Interactive plots (Plotly)
   - Real-time training dashboards
   - Comparison reports (HTML/PDF)

7. **Optimization**
   - Flash Attention integration
   - Mixed precision training
   - Gradient checkpointing options

### Nice to Have

8. **Documentation**
   - Video tutorials
   - More examples
   - Architecture diagrams

9. **Tools**
   - Docker containers
   - Model serving scripts
   - API endpoints

10. **CI/CD**
    - GitHub Actions workflows
    - Automated testing
    - Documentation generation

## Project Structure

```
budget-finetune-pipeline-qlora-spectrum/
├── src/
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation and visualization
│   └── utils/             # Utility modules
├── configs/               # Configuration files
├── results/               # Output directory
├── tests/                 # Test files (to be added)
├── docs/                  # Additional documentation (to be added)
└── examples/              # Example scripts (to be added)
```

## Adding a New Training Method

1. **Create training script**: `src/training/train_yourmethod.py`
2. **Implement adapter/modification**: Follow QLoRA or Spectrum as template
3. **Add configuration**: `configs/yourmethod_config.yaml`
4. **Update benchmark runner**: Add method to `run_benchmark.py`
5. **Add tests**: Create `tests/test_yourmethod.py`
6. **Update documentation**: Add to README.md and COMPARISON.md

## Code Review Process

1. **Automated Checks**: Code must pass linting and tests
2. **Manual Review**: Maintainer will review code quality and design
3. **Testing**: Verify functionality on actual hardware
4. **Documentation**: Ensure docs are updated
5. **Merge**: Approved PRs will be merged

## Commit Message Guidelines

Use conventional commits format:

```
<type>: <description>

[optional body]
[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat: Add DoRA training method
fix: Resolve CUDA out of memory error in QLoRA
docs: Update USAGE.md with new parameters
refactor: Simplify cost calculation logic
test: Add unit tests for MemoryTracker
```

## Performance Benchmarking

When adding new methods:

1. **Run benchmark**: Use `run_benchmark.py`
2. **Collect metrics**: Cost, time, memory, quality
3. **Compare**: Against existing methods
4. **Document**: Add to COMPARISON.md

## Questions?

- Open an issue for questions
- Tag with `question` label
- Check existing issues first

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use GitHub Issues for bugs and features
- 📧 **Email**: For private matters only

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort! 🎉

---

**Happy Contributing! 🚀**
