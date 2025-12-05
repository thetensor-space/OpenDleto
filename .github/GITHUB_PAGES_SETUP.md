# GitHub Pages Setup Instructions

This repository is configured to use GitHub Pages to host documentation from the `/docs` folder on a dedicated `www` branch.

## Overview

- **Branch**: `www`
- **Content Source**: `/docs` folder
- **Site URL** (once enabled): `https://thetensor-space.github.io/OpenDleto/`

## Initial Setup Steps

### 1. Create the www Branch

The `www` branch needs to be created with the repository content and the GitHub Pages `index.html` file. You can do this in one of two ways:

#### Option A: Using the GitHub Actions Workflow (Recommended)

1. Go to the **Actions** tab in the GitHub repository
2. Select the "Update www branch for GitHub Pages" workflow
3. Click "Run workflow" button
4. Select the branch to run from (typically `main`)
5. Click "Run workflow"

This will automatically create the `www` branch if it doesn't exist and add the required `index.html` file.

#### Option B: Manual Branch Creation

If you have repository write access, you can create the branch manually:

```bash
# Clone the repository
git clone https://github.com/thetensor-space/OpenDleto.git
cd OpenDleto

# Create and checkout the www branch
git checkout -b www

# The index.html should already exist in docs/ from the PR
# If not, you can create it or merge from the setup branch

# Push the branch to remote
git push -u origin www
```

### 2. Enable GitHub Pages in Repository Settings

Once the `www` branch exists on GitHub:

1. Navigate to the repository **Settings**
2. In the left sidebar, click **Pages** (under "Code and automation")
3. Under **Source**:
   - Select **Deploy from a branch**
4. Under **Branch**:
   - Select `www` from the branch dropdown
   - Select `/docs` from the folder dropdown
   - Click **Save**

### 3. Wait for Deployment

- GitHub will automatically build and deploy the site
- This usually takes 1-2 minutes
- You'll see a green checkmark when deployment is complete
- The site will be available at: `https://thetensor-space.github.io/OpenDleto/`

## Maintaining the www Branch

### Automatic Updates

The GitHub Actions workflow (`.github/workflows/update-www-branch.yml`) is configured to:
- Automatically update the `www` branch when changes are pushed to `main` branch that affect the `docs/` directory
- Can be manually triggered from the Actions tab

### Manual Updates

To manually update the www branch with changes from main:

```bash
# Checkout www branch
git checkout www

# Merge or rebase from main
git merge main
# or
git rebase main

# Push changes
git push origin www
```

## File Structure

```
OpenDleto/
├── docs/
│   ├── index.html          # GitHub Pages landing page (required)
│   ├── Installing-Dleto.md # Documentation
│   └── images/             # Image assets
├── .github/
│   ├── workflows/
│   │   └── update-www-branch.yml  # Automation workflow
│   └── GITHUB_PAGES_SETUP.md      # This file
└── README.md               # Includes GitHub Pages setup instructions
```

## Troubleshooting

### The www branch doesn't exist
- Run the GitHub Actions workflow as described in "Option A" above
- Or create it manually as described in "Option B"

### Pages not deploying
- Check that GitHub Pages is enabled in Settings > Pages
- Ensure the `www` branch is selected with `/docs` folder
- Check the Actions tab for any deployment errors

### 404 Error when visiting the site
- Verify the `docs/index.html` file exists on the `www` branch
- Ensure GitHub Pages is configured correctly in Settings
- Wait a few minutes for GitHub's CDN to update

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [OpenDleto Repository](https://github.com/thetensor-space/OpenDleto)
- [TheTensor.Space](https://TheTensor.Space/)
