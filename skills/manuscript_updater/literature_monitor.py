#!/usr/bin/env python3
"""
Literature surveillance tool for identifying relevant research papers
in mechanistic chemistry and AI for manuscript integration.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
from urllib.parse import quote


class LiteratureMonitor:
    """Monitor academic literature for manuscript-relevant papers."""

    def __init__(self):
        self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1"
        self.pubmed_api = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.arxiv_api = "https://export.arxiv.org/api/query"

        # Keywords relevant to mechanistic chemistry and AI
        self.keywords = [
            "reaction mechanism prediction",
            "organic reaction mechanisms AI",
            "mechanistic chemistry machine learning",
            "chemical reaction networks",
            "retrosynthesis AI",
            "reaction pathway prediction",
            "computational mechanistic chemistry",
            "AI chemical synthesis planning",
            "machine learning reaction mechanisms",
            "deep learning organic chemistry"
        ]

    def search_recent_papers(self, months_back: int = 6) -> List[Dict]:
        """Search for recent papers in relevant areas."""
        papers = []
        since_date = (datetime.now() - timedelta(days=30*months_back)).strftime("%Y-%m-%d")

        for keyword in self.keywords:
            try:
                # Search Semantic Scholar
                ss_papers = self._search_semantic_scholar(keyword, since_date)
                papers.extend(ss_papers)

                # Search arXiv
                arxiv_papers = self._search_arxiv(keyword, months_back)
                papers.extend(arxiv_papers)

                time.sleep(1)  # Rate limiting

            except Exception as e:
                print(f"Error searching for '{keyword}': {e}")

        # Remove duplicates and sort by relevance/date
        unique_papers = self._deduplicate_papers(papers)
        return sorted(unique_papers, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _search_semantic_scholar(self, query: str, since_date: str) -> List[Dict]:
        """Search Semantic Scholar for relevant papers."""
        papers = []

        # Build search query
        search_query = f"{query} AND publicationDate:{since_date}:*"

        url = f"{self.semantic_scholar_api}/paper/search"
        params = {
            'query': search_query,
            'limit': 20,
            'fields': 'title,authors,abstract,year,venue,citationCount,influentialCitationCount,externalIds'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for paper in data.get('data', []):
                processed_paper = {
                    'title': paper.get('title', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'abstract': paper.get('abstract', ''),
                    'year': paper.get('year', ''),
                    'venue': paper.get('venue', ''),
                    'citations': paper.get('citationCount', 0),
                    'influential_citations': paper.get('influentialCitationCount', 0),
                    'doi': paper.get('externalIds', {}).get('DOI', ''),
                    'source': 'Semantic Scholar',
                    'relevance_score': self._calculate_relevance(paper, query),
                    'manuscript_relevance': self._assess_manuscript_relevance(paper)
                }
                papers.append(processed_paper)

        except Exception as e:
            print(f"Semantic Scholar search failed: {e}")

        return papers

    def _search_arxiv(self, query: str, months_back: int) -> List[Dict]:
        """Search arXiv for recent preprints."""
        papers = []

        url = self.arxiv_api
        params = {
            'search_query': f'all:{quote(query)}',
            'start': 0,
            'max_results': 20,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Parse XML response (simplified - would need proper XML parsing)
            entries = re.findall(r'<entry>.*?</entry>', response.text, re.DOTALL)

            for entry in entries:
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                authors_match = re.findall(r'<name>(.*?)</name>', entry)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                published_match = re.search(r'<published>(.*?)</published>', entry)

                if title_match:
                    # Check if recent enough
                    if published_match:
                        pub_date = datetime.fromisoformat(published_match.group(1)[:19])
                        if pub_date < (datetime.now() - timedelta(days=30*months_back)):
                            continue

                    paper = {
                        'title': re.sub(r'<[^>]+>', '', title_match.group(1)).strip(),
                        'authors': [re.sub(r'<[^>]+>', '', author).strip() for author in authors_match],
                        'abstract': re.sub(r'<[^>]+>', '', summary_match.group(1)).strip() if summary_match else '',
                        'year': pub_date.year if published_match else datetime.now().year,
                        'venue': 'arXiv',
                        'citations': 0,  # arXiv doesn't have citations
                        'influential_citations': 0,
                        'doi': '',  # Would need to extract arXiv ID
                        'source': 'arXiv',
                        'relevance_score': self._calculate_arxiv_relevance(entry, query),
                        'manuscript_relevance': 'medium'  # arXiv papers are often cutting-edge
                    }
                    papers.append(paper)

        except Exception as e:
            print(f"arXiv search failed: {e}")

        return papers

    def _calculate_relevance(self, paper: Dict, query: str) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0

        # Citation-based scoring
        score += min(paper.get('citationCount', 0) / 100, 2.0)

        # Influential citation bonus
        score += min(paper.get('influentialCitationCount', 0) / 20, 1.0)

        # Keyword matching in title/abstract
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        query_terms = query.lower().split()

        matches = sum(1 for term in query_terms if term in text)
        score += matches * 0.5

        # Venue prestige (simplified)
        prestige_venues = ['nature', 'science', 'jacs', 'angewandte', 'chemrxiv']
        venue = paper.get('venue', '').lower()
        if any(pv in venue for pv in prestige_venues):
            score += 1.0

        return score

    def _calculate_arxiv_relevance(self, entry_xml: str, query: str) -> float:
        """Calculate relevance for arXiv papers."""
        score = 0.5  # Base score for arXiv (cutting-edge work)

        # Keyword matching
        text = entry_xml.lower()
        query_terms = query.lower().split()

        matches = sum(1 for term in query_terms if term in text)
        score += matches * 0.3

        return score

    def _assess_manuscript_relevance(self, paper: Dict) -> str:
        """Assess how relevant this paper is to the Mechanistic manuscript."""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        # High relevance indicators
        high_terms = [
            'mechanistic', 'reaction mechanism', 'organic synthesis',
            'computational chemistry', 'machine learning chemistry',
            'ai chemistry', 'reaction prediction', 'synthesis planning'
        ]

        # Medium relevance indicators
        medium_terms = [
            'chemistry ai', 'chemical ai', 'molecular machine learning',
            'reaction classification', 'synthesis', 'organic chemistry ml'
        ]

        if any(term in text for term in high_terms):
            return 'high'
        elif any(term in text for term in medium_terms):
            return 'medium'
        else:
            return 'low'

    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            # Normalize title for comparison
            title_norm = re.sub(r'[^\\w\\s]', '', paper['title'].lower())
            title_norm = ' '.join(title_norm.split())  # Normalize whitespace

            if title_norm not in seen_titles:
                seen_titles.add(title_norm)
                unique_papers.append(paper)

        return unique_papers

    def generate_citation_suggestions(self, papers: List[Dict]) -> List[Dict]:
        """Generate manuscript citation suggestions for relevant papers."""
        suggestions = []

        for paper in papers:
            if paper.get('manuscript_relevance') in ['high', 'medium']:
                suggestion = {
                    'paper': paper,
                    'suggested_sections': self._suggest_sections(paper),
                    'citation_text': self._generate_citation_text(paper),
                    'discussion_points': self._generate_discussion_points(paper)
                }
                suggestions.append(suggestion)

        return suggestions

    def _suggest_sections(self, paper: Dict) -> List[str]:
        """Suggest manuscript sections where this paper should be cited."""
        suggestions = []
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        if 'method' in text or 'approach' in text or 'algorithm' in text:
            suggestions.append('Methods')
        if 'result' in text or 'performance' in text or 'evaluation' in text:
            suggestions.append('Results')
        if 'related' in text or 'previous' in text or 'literature' in text:
            suggestions.append('Related Work')
        if 'future' in text or 'limitation' in text or 'challenge' in text:
            suggestions.append('Discussion')

        # Default sections if no specific matches
        if not suggestions:
            suggestions = ['Related Work', 'Discussion']

        return suggestions

    def _generate_citation_text(self, paper: Dict) -> str:
        """Generate APA-style citation text."""
        authors = paper.get('authors', [])
        year = paper.get('year', '')
        title = paper.get('title', '')

        if len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_text = f"{authors[0]} et al."
        else:
            author_text = "Unknown Authors"

        return f"{author_text} ({year}). {title}."

    def _generate_discussion_points(self, paper: Dict) -> List[str]:
        """Generate discussion points for manuscript integration."""
        points = []
        text = paper.get('abstract', '').lower()

        # Extract key findings or contributions
        if 'we show' in text or 'we demonstrate' in text:
            points.append("Demonstrates novel approach to reaction mechanism prediction")

        if 'improve' in text or 'better' in text or 'superior' in text:
            points.append("Shows performance improvements over existing methods")

        if 'challenge' in text or 'limitation' in text:
            points.append("Addresses key challenges in computational mechanistic chemistry")

        if 'future' in text or 'direction' in text:
            points.append("Suggests promising directions for future research")

        # Default point if nothing specific found
        if not points:
            points.append("Contributes to the growing literature on AI-assisted chemistry")

        return points


def main():
    """Command-line interface for testing the literature monitor."""
    monitor = LiteratureMonitor()

    print("Searching for recent relevant papers...")
    papers = monitor.search_recent_papers(months_back=3)

    print(f"\nFound {len(papers)} relevant papers")

    for i, paper in enumerate(papers[:5]):  # Show top 5
        print(f"\n{i+1}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}")
        print(f"   Year: {paper['year']}, Venue: {paper['venue']}")
        print(f"   Relevance: {paper['manuscript_relevance']}, Score: {paper['relevance_score']:.2f}")
        print(f"   Citations: {paper['citations']}")

    print("\nGenerating citation suggestions...")
    suggestions = monitor.generate_citation_suggestions(papers[:3])

    for suggestion in suggestions:
        paper = suggestion['paper']
        print(f"\nCitation: {suggestion['citation_text']}")
        print(f"Sections: {', '.join(suggestion['suggested_sections'])}")
        print(f"Discussion: {suggestion['discussion_points'][0]}")


if __name__ == '__main__':
    main()

