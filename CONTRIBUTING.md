# Contributing to `pythTB`

Thank you for your interest in contributing to `pythTB`! Your contributions help make the project more useful and accessible for everyone. Please take a moment to review the guidelines below before submitting changes.

## Code Quality and Design
### Clarity and maintainability
Favor clear, straightforward code over clever but opaque solutions. Aim for reliability and long-term readability. This includes being thoughtful when naming variables and functions; names should reflect their purpose.
### Performance
Use vectorized operations where practical to reduce bottlenecks. Minimize deeply nested loops, especially in linear-algebra-heavy routines.
### Documentation
Document all non-obvious behavior, conventions, and edge cases. Use comments in code and update the docs so others can easily understand and build on your work.
### Class and API design
Keep interfaces minimal and intuitive. Avoid unnecessary complexity; simple, well-structured classes are easier to extend and maintain.
### Ambiguity
If a function could return confusing or misleading results, prefer raising a warning or leaving the function private rather than returning something ambiguous to the user.

## How to Contribute
1. Fork the repository on GitHub.
2. Create a feature or bugfix branch with a clear name (e.g. feature/add-new-solver, bugfix/fix-edge-case-handling).
3. Write and update tests for any new functionality. Cover edge cases to ensure reliability.
4. Update documentation and docstrings to reflect your changes.
5. Open a pull request with a clear description of your changes and their motivation.

## Reporting Issues

If you run into bugs or have ideas for improvements:
- Open an issue on GitHub.
- Include relevant details: steps to reproduce, error messages, minimal examples, and system information.

## Code Reviews

All pull requests are reviewed by maintainers. Feedback and iteration help keep the codebase consistent and high-quality.

## Final Note

Our goal is to make *PythTB* clear, robust, and welcoming to contributors. Thoughtful code, careful documentation, and clear communication all help the project grow in a sustainable way.
