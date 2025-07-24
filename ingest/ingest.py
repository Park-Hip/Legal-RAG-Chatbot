from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import re
from datetime import datetime
from langchain.schema import Document
import pickle
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
)
logger = logging.getLogger(__name__)

loader = DirectoryLoader(
    path='../',
    glob='**/*.pdf',
    loader_cls=PyMuPDFLoader
)
docs = loader.load()

chapter_pattern = r'\n(Chapter\s*[A-Z ,;]+)\s*\n([A-Z ]+)'
doc_list = []
full_text = ''
title = docs[0].metadata['title']
source = docs[0].metadata['source']
article_num = 0

for doc in docs:
    doc_text = doc.page_content
    full_text += doc_text
    page_num = doc.metadata['page']
chapters = list(re.finditer(chapter_pattern, full_text))

for i, chapter in enumerate(chapters):
    chapter_start = chapters[i].start()
    chapter_end = chapters[i + 1].start() if i + 1 < len(chapters) else len(full_text)
    chapter_text = full_text[chapter_start:chapter_end]
    logger.info(chapters[i].group().strip())

    article_pattern = r'(Article\s*\d+\.\s+[a-zA-Z ,;]+)[^\s\n]+'
    articles = list(re.finditer(article_pattern, chapter_text))
    article_num += len(articles)
    logger.info(f'Total: {len(articles)} articles')

    for j, article in enumerate(articles):
        article_start = articles[j].start()
        article_end = articles[j + 1].start() if j + 1 < len(articles) else chapter_end
        article_text = chapter_text[article_start:article_end]
        doc_list.append(Document(
            page_content=article_text,
            metadata={
                'chapter': chapters[i].group().replace('\n', '').strip(),
                'article': articles[j].group().replace('\n', '').strip(),
                'id': str(uuid.uuid4()),
                'title': title,
                'source': source,
                'timestamp_now': datetime.now().isoformat(),
                'word_count': len(article_text.split()),
                'token_count': len(article_text)
            }))
        logger.info(f'  {articles[j].group()}')

    logger.info('-----------------------------------------------------')
logger.info(f'{article_num} articles stored!')

with open('../data/processed_data/criminal_code_of_vietnam.pkl', 'wb') as f:
    pickle.dump(doc_list,f)
logger.info('Doc list has been created')
