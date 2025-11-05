# Contributing to `PythTB`

Thank you for your interest in contributing to `PythTB`! Your contributions help make the project more useful and accessible for everyone. Please take a moment to review the guidelines below before submitting changes.

## Code Quality and Design
### Clarity and maintainability
Favor clear, straightforward code over clever but opaque solutions. Aim for reliability and long-term readability. This includes being thoughtful when naming variables and functions; names should reflect their purpose and follow Python conventions.
### Performance
Use vectorized operations where practical to reduce bottlenecks. Minimize deeply nested loops, especially in linear-algebra-heavy routines.
### Documentation
Document all non-obvious behavior, conventions, and edge cases. Use comments in your code and update the documentation so that others can easily understand and build upon your work.
### Class and API design
Keep interfaces minimal and intuitive. Avoid unnecessary complexity; simple, well-structured classes are easier to extend and maintain.
### Ambiguity
If a function could return confusing or misleading results, prefer raising a warning or leaving the function private rather than returning something ambiguous to the user.
### Code Reviews
All pull requests are reviewed by maintainers. Feedback and iteration help maintain a consistent and high-quality codebase.

## How to Contribute
1. For first-time contributors:
   - Click the "fork" button to create your own copy of the project.
   - Clone the project to your local computer:
     ```bash
     git clone https://github.com/your-username/pythtb.git
     ```
    - Add the upstream repository:
      ```bash
      git remote add upstream https://github.com/pythtb/pythtb.git
      ```
    - Now, `git remote -v` will show two remote repositories named:
      - `upstream` referring to the `pythtb` repository
      - `origin`, which refers to your personal fork
    - Pull the latest changes from upstream, including tags:
      ```bash
      git checkout main
      git pull upstream main --tags
      ```
2. Develop your own contributions:
   - Create a branch for the feature you want to work on. Use something related to the feature, such as 'wfarray-speedups'
     ```bash
     git checkout -b wfarray-speedups
     ```
   - Make frequent commits locally as you implement changes.
   - Be sure to document any changed behavior in docstrings,
     using [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) conventions.
   - Add tests for new functionality. Resolve any failing tests before pushing.
   - If the Sphinx webpage has been updated, build locally to make sure everything renders correctly and that there are no warnings or errors. 
3. Submit your contributions:
   - Push changes back to your fork on GitHub:
     ```bash
     git push origin wfarray-speedups
     ```
   - Go to GitHub, the new branch will show up with a green "Pull Request" (PR) button. Click it. Make sure the title and message
     are descriptive and self-explanatory. Then click to submit.
4. Review process:
   - Reviewers will write comments back on your PR to address implementation, documentation, or style concerns.
   - Once the PR has been approved by at least one of the maintainers, the PR is ready to be merged.
5. Document changes:
   - If the changes are user-facing, they may need to be added to the [CHANGELOG](CHANGELOG.md). See [keep a changelog](https://keepachangelog.com/en/1.1.0/)
     for the layout. 

## Reporting Issues

If you run into bugs or have ideas for improvements:
- Open an [issue on GitHub](https://github.com/pythtb/PythTB/issues).
- Include relevant details: steps to reproduce, error messages, minimal examples, and system information.
