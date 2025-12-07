#!/usr/bin/env python3
"""
Autonomous Documentation Downloader

Downloads and saves official documentation for key frameworks:
- FastAPI
- Docker
- Pydantic
- scikit-learn
- Uvicorn
- pytest

Saves to Reference_Docs/ with proper organization.
"""

import os
import sys
import requests
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
import json

# Target frameworks and their documentation
FRAMEWORKS = {
    "FastAPI": {
        "base_url": "https://fastapi.tiangolo.com",
        "pages": [
            "/",
            "/intro/",
            "/features/",
            "/tutorial/",
            "/tutorial/first-steps/",
            "/tutorial/body/",
            "/tutorial/body-updates/",
            "/tutorial/query-parameters/",
            "/tutorial/request-body/",
            "/tutorial/response-model/",
            "/tutorial/errors/",
            "/advanced/",
            "/deployment/",
            "/deployment/concepts/",
        ],
        "folder": "FastAPI_Documentation"
    },
    
    "Docker": {
        "base_url": "https://docs.docker.com",
        "pages": [
            "/get-started/",
            "/get-started/overview/",
            "/build/building/dockerfile/",
            "/build/building/best-practices/",
            "/compose/gettingstarted/",
            "/engine/reference/",
            "/develop/dev-best-practices/",
        ],
        "folder": "Docker_Documentation"
    },
    
    "Pydantic": {
        "base_url": "https://docs.pydantic.dev",
        "pages": [
            "/",
            "/latest/",
            "/latest/getting-started/",
            "/latest/concepts/models/",
            "/latest/concepts/validators/",
            "/latest/concepts/fields/",
            "/latest/api/fields/",
        ],
        "folder": "Pydantic_Documentation"
    },
    
    "scikit-learn": {
        "base_url": "https://scikit-learn.org/stable",
        "pages": [
            "/",
            "/modules/classes/",
            "/modules/model_evaluation.html",
            "/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
            "/modules/generated/sklearn.preprocessing.RobustScaler.html",
            "/modules/generated/sklearn.pipeline.Pipeline.html",
            "/modules/generated/sklearn.metrics.r2_score.html",
            "/modules/generated/sklearn.metrics.mean_absolute_error.html",
            "/modules/generated/sklearn.metrics.mean_squared_error.html",
        ],
        "folder": "Scikit-Learn_Documentation"
    },
    
    "Uvicorn": {
        "base_url": "https://www.uvicorn.org",
        "pages": [
            "/",
            "/deployment/",
            "/settings/",
        ],
        "folder": "Uvicorn_Documentation"
    },
    
    "pytest": {
        "base_url": "https://docs.pytest.org/en/stable",
        "pages": [
            "/",
            "/how-to/index.html",
            "/reference/",
            "/getting-started.html",
        ],
        "folder": "pytest_Documentation"
    }
}

class DocumentationDownloader:
    def __init__(self, base_dir="Reference_Docs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Real-Estate-Estimator-DocDownloader/1.0'
        })
        self.downloaded = []
        self.failed = []
        
    def download_page(self, url, framework_folder):
        """Download a single page and save as markdown."""
        try:
            print(f"  Downloading: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract filename from URL
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            filename = path.split('/')[-1] or 'index'
            
            if filename.endswith('.html'):
                filename = filename[:-5]
            
            filename = filename.replace('-', '_').replace('/', '_') + '.md'
            
            # Save content
            folder_path = self.base_dir / framework_folder
            folder_path.mkdir(exist_ok=True)
            
            file_path = folder_path / filename
            
            # Create markdown with metadata
            markdown_content = f"""# {filename.replace('_', ' ')}

**Source:** {url}  
**Downloaded:** {datetime.now().isoformat()}  
**Framework:** {framework_folder.replace('_Documentation', '')}  

---

## Content

The original HTML documentation has been downloaded. Key information:

```html
{response.text[:5000]}  # First 5000 chars
...
(Full content available at source URL)
```

**Full Documentation:** [View at {url}]({url})

---

**Note:** This is a snapshot of the official documentation.
For the latest and most complete information, visit the original URL above.
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self.downloaded.append((framework_folder, filename))
            print(f"    ✓ Saved: {file_path}")
            time.sleep(0.5)  # Be polite to servers
            
        except Exception as e:
            error_msg = f"{framework_folder}/{filename}: {str(e)}"
            self.failed.append(error_msg)
            print(f"    ✗ Failed: {error_msg}")
    
    def download_framework(self, framework_name, config):
        """Download all pages for a framework."""
        print(f"\nDownloading {framework_name}...")
        print(f"  Base URL: {config['base_url']}")
        print(f"  Target folder: {config['folder']}")
        
        for page_path in config['pages']:
            url = urljoin(config['base_url'], page_path)
            self.download_page(url, config['folder'])
    
    def create_framework_index(self, framework_name, config):
        """Create index file for framework documentation."""
        folder_path = self.base_dir / config['folder']
        index_file = folder_path / "README.md"
        
        # List downloaded files
        files = list(folder_path.glob("*.md"))
        files.sort()
        
        index_content = f"""# {framework_name} Documentation

**Downloaded:** {datetime.now().isoformat()}  
**Source:** {config['base_url']}  

## Available Documentation

| File | Purpose |
|------|---------|
"""
        
        for f in files:
            if f.name != "README.md":
                index_content += f"| [{f.stem}]({f.name}) | Documentation page |\n"
        
        index_content += f"""

## Quick Links

- **Official Documentation:** [{config['base_url']}]({config['base_url']})
- **API Reference:** Check individual pages

## Keywords

- {framework_name}
- API
- Reference
- Documentation

---

**Note:** This documentation was automatically downloaded for offline reference.
For the latest documentation, visit the official website above.
"""
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        print(f"  ✓ Created index: {index_file}")
    
    def run(self):
        """Run downloader for all frameworks."""
        print("=" * 60)
        print("DOCUMENTATION DOWNLOADER")
        print(f"Downloading documentation to: {self.base_dir}")
        print("=" * 60)
        
        for framework_name, config in FRAMEWORKS.items():
            try:
                self.download_framework(framework_name, config)
                self.create_framework_index(framework_name, config)
            except Exception as e:
                print(f"\n✗ Error downloading {framework_name}: {e}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print download summary."""
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        
        print(f"\n✓ Successfully downloaded: {len(self.downloaded)} pages")
        if self.downloaded:
            for framework, filename in self.downloaded[:10]:
                print(f"  - {framework}: {filename}")
            if len(self.downloaded) > 10:
                print(f"  ... and {len(self.downloaded) - 10} more")
        
        print(f"\n✗ Failed: {len(self.failed)} pages")
        if self.failed:
            for error in self.failed[:5]:
                print(f"  - {error}")
            if len(self.failed) > 5:
                print(f"  ... and {len(self.failed) - 5} more")
        
        print(f"\nDocumentation saved to: {self.base_dir}")
        print(f"Next steps:")
        print(f"  1. Review downloaded documentation")
        print(f"  2. Create master index (master_docs_index.md)")
        print(f"  3. Update search index for quick lookup")
        print("=" * 60)


def main():
    """Main entry point."""
    # Determine base directory
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default to Reference_Docs in project root
        base_dir = "Reference_Docs"
    
    downloader = DocumentationDownloader(base_dir)
    downloader.run()


if __name__ == "__main__":
    main()





