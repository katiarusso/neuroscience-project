# How to Sync Your Local Repo
> The history of this repo was recently rewritten to remove large files.
> Follow these steps **once** to get back in sync. You won't lose any of your own work if you follow carefully.

## Step 1 — Save any uncommitted work
Before doing anything, make sure you haven't lost any unsaved changes.
```bash
git status
```
If you see modified files, stash them temporarily:
```bash
git stash
```
## Step 2 — Fetch the latest history from GitHub
```bash
git fetch --all
```
## Step 3 — Reset your local branch to match GitHub
```bash
git reset --hard origin/main
```
> This will update your local `main` to exactly match the remote.
> Don't worry — your files on disk (notebooks, data, etc.) are not deleted.

## Step 4 — Restore your stashed work (if you stashed in Step 1)
```bash
git stash pop
```
## Step 5 — Verify everything is good
```bash
git log --oneline -5
git status
```
You should see the latest commits and a clean working tree.

## Important: `.pkl` files are now ignored
Large `.pkl` files (model outputs, data arrays, etc.) are **no longer tracked by git**.
They live only on your local machine.
If you need to share data files with the team, use a shared drive or cloud storage instead.

## Need help?
Ask Tommy.
