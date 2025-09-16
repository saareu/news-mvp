from setuptools import setup, find_packages

setup(
    name="news-etl",
    version="0.1.0",
    description="ETL pipeline for Israeli news sources (Ynet, Hayom, Haaretz)",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.25",
        "feedparser>=6.0",
        "selectolax>=0.3.12",
        "python-dateutil>=2.9",
        "tzdata>=2024.1; platform_system == 'Windows'",
        "openpyxl>=3.1",
        "beautifulsoup4>=4.12",
        "xmltodict>=0.13",
        "requests>=2.31.0",
        "pandas>=2.0",
        "plotly>=5.0",
        "langdetect==1.0.9",
        "pyyaml>=6.0"
    ],
    entry_points={
        "console_scripts": [
            "news-etl=etl.cli:main"
        ]
    },
    python_requires=">=3.8",
    include_package_data=True,
)
