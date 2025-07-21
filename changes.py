import subprocess
import os
import sys
from datetime import datetime

def run_git_command(command):
    """Run a git command and return the output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd(),
            encoding='utf-8',
            errors='replace'  # Replace problematic characters instead of failing
        )
        if result.returncode != 0:
            print(f"Error running git command: {command}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Exception running command: {e}")
        return None

def check_git_repo():
    """Check if current directory is a git repository"""
    if not os.path.exists('.git'):
        print("Error: Current directory is not a git repository")
        sys.exit(1)

def check_commit_exists(commit_hash):
    """Check if a commit exists in the repository"""
    command = f'git cat-file -e {commit_hash}'
    result = subprocess.run(command, shell=True, capture_output=True, cwd=os.getcwd())
    return result.returncode == 0

def get_commit_details(commit_hash):
    """Get detailed information about a specific commit"""
    command = f'git show --stat --pretty=fuller {commit_hash}'
    return run_git_command(command)

def get_commit_diff(commit_hash):
    """Get the full diff for a specific commit"""
    command = f'git show --pretty="" {commit_hash}'
    return run_git_command(command)

def get_commits_in_range(first_commit, last_commit):
    """Get all commits between first and last commit (inclusive)"""
    # First check if both commits exist
    if not check_commit_exists(first_commit):
        print(f"Error: First commit {first_commit} does not exist in this repository")
        return []
    
    if not check_commit_exists(last_commit):
        print(f"Error: Last commit {last_commit} does not exist in this repository")
        return []
    
    # Try different approaches to get the commit range
    commands_to_try = [
        f'git rev-list --reverse {first_commit}..{last_commit}',  # Excludes first commit
        f'git rev-list --reverse {first_commit}^..{last_commit}', # Includes first commit
        f'git log --reverse --pretty=format:"%H" {first_commit}..{last_commit}',
        f'git log --reverse --pretty=format:"%H" {first_commit}^..{last_commit}'
    ]
    
    for i, command in enumerate(commands_to_try):
        print(f"Trying method {i+1}: {command}")
        result = run_git_command(command)
        if result and result.strip():
            commits = result.strip().split('\n')
            print(f"Found {len(commits)} commits with method {i+1}")
            return commits
    
    # If no range works, maybe they're the same commit or reversed
    print("No commits found in range. Checking if commits are the same or in reverse order...")
    
    # Check if first_commit == last_commit
    if first_commit == last_commit:
        print("First and last commits are the same, returning single commit")
        return [first_commit]
    
    # Try reverse order
    command = f'git rev-list --reverse {last_commit}..{first_commit}'
    result = run_git_command(command)
    if result and result.strip():
        commits = result.strip().split('\n')
        print(f"Found {len(commits)} commits in reverse order")
        return commits
    
    return []

def get_files_changed(commit_hash):
    """Get list of files changed in a commit"""
    command = f'git diff-tree --no-commit-id --name-only -r {commit_hash}'
    result = run_git_command(command)
    if result:
        return [f for f in result.strip().split('\n') if f]
    return []

def main():
    # Commit IDs
    first_commit = "416e65bf172876311b91362f35d02690ebce2418"
    last_commit = "d3e1e537460ceb0c445aeb5f25a05e09ed6913c7"
    
    # Check if we're in a git repository
    check_git_repo()
    
    print("Generating detailed changes report...")
    print(f"Looking for commits between {first_commit[:8]} and {last_commit[:8]}")
    
    # Get all commits in the range
    commits = get_commits_in_range(first_commit, last_commit)
    
    if not commits:
        print("No commits found in the specified range.")
        print("Let me try to get information about both individual commits...")
        
        # If no range found, at least process the individual commits
        commits = []
        if check_commit_exists(first_commit):
            commits.append(first_commit)
        if check_commit_exists(last_commit) and last_commit != first_commit:
            commits.append(last_commit)
        
        if not commits:
            print("Error: Neither commit exists in this repository")
            sys.exit(1)
    
    print(f"Processing {len(commits)} commit(s)...")
    
    # Create the changes.txt file with UTF-8 encoding
    with open('changes.txt', 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("ULTRA DETAILED GIT CHANGES REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Repository: {os.getcwd()}\n")
        f.write(f"First commit: {first_commit}\n")
        f.write(f"Last commit: {last_commit}\n")
        f.write(f"Total commits in range: {len(commits)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Process each commit
        for i, commit in enumerate(commits, 1):
            print(f"Processing commit {i}/{len(commits)}: {commit[:8]}")
            
            f.write(f"\n{'#' * 60}\n")
            f.write(f"COMMIT {i} OF {len(commits)}\n")
            f.write(f"{'#' * 60}\n\n")
            
            # Get commit details
            commit_details = get_commit_details(commit)
            if commit_details:
                f.write("COMMIT INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(commit_details)
                f.write("\n")
            
            # Get files changed
            files_changed = get_files_changed(commit)
            if files_changed:
                f.write("FILES CHANGED:\n")
                f.write("-" * 40 + "\n")
                for file_path in files_changed:
                    f.write(f"  - {file_path}\n")
                f.write("\n")
            
            # Get full diff
            f.write("DETAILED CHANGES (DIFF):\n")
            f.write("-" * 40 + "\n")
            commit_diff = get_commit_diff(commit)
            if commit_diff:
                f.write(commit_diff)
            else:
                f.write("No changes or error retrieving diff\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Write summary at the end
        f.write("\n\nSUMMARY:\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total commits processed: {len(commits)}\n")
        f.write(f"Report generated successfully!\n")
    
    print(f"\nReport generated successfully! Check 'changes.txt' for ultra-detailed changes.")
    print(f"Total commits processed: {len(commits)}")

if __name__ == "__main__":
    main()