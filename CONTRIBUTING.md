# Contributing to Enterprise Document Intelligence

Thank you for your interest in contributing! 🎉

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/enterprise-document-intelligence.git
   cd enterprise-document-intelligence
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise code
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Include type hints where appropriate

3. **Run tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

4. **Format your code**
   ```bash
   black src/ tests/ examples/
   flake8 src/ tests/ examples/ --max-line-length=100
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   Use conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/modifications
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub

## Code Style

- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 100 characters
- Use type hints where applicable
- Write descriptive variable and function names

## Testing

- Write unit tests for new features
- Maintain or improve code coverage
- Tests should be self-contained and fast
- Use fixtures for common test setup

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions/classes
- Include usage examples for new features
- Update config.yaml comments if adding new settings

## Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what and why
- **Tests**: Include relevant tests
- **Documentation**: Update as needed
- **One feature per PR**: Keep PRs focused

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

Thank you for contributing! 🚀
