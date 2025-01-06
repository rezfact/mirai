import aiohttp
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.database import StockInfo
import logging
import ssl

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 25  # seconds

def parse_indonesian_date(date_string):
    if date_string == '-':
        return None
    month_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    day, month, year = date_string.split()
    return datetime(int(year), month_map[month], int(day))

async def fetch_with_retry(session, url, retries=MAX_RETRIES):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    for attempt in range(retries):
        try:
            async with session.get(url, ssl=ssl_context) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for URL: {url}")
        except aiohttp.ClientError as e:
            logger.error(f"Attempt {attempt + 1} failed for URL: {url}. Error: {str(e)}")
        
        if attempt < retries - 1:
            await asyncio.sleep(RETRY_DELAY)
    
    raise Exception(f"Failed to fetch data from {url} after {retries} attempts")

async def fetch_ksei_data(month: int, year: int):
    url = f"https://www.ksei.co.id/ksei_calendar/get_json/event-{month}-{year}-all.json"
    async with aiohttp.ClientSession() as session:
        data = await fetch_with_retry(session, url)
        return await asyncio.get_event_loop().run_in_executor(None, lambda: eval(data))

async def fetch_ksei_detail(url: str):
    async with aiohttp.ClientSession() as session:
        return await fetch_with_retry(session, url)

async def parse_ksei_detail(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    data = {}
    
    data['event_type'] = soup.select_one('h2.accordion__title').text.strip()
    data['event_subtype'] = soup.select_one('section.accordion--secondary > h2.accordion__title').text.strip()
    data['stock_name'] = soup.select_one('section.accordion--last > h2.accordion__title').text.strip()

    dl = soup.select_one('dl.accordion-dl')
    for dt in dl.select('dt'):
        if dt.text.strip() == 'Security Detail':
            data['security_code'] = dt.find_next('span').text.strip()
            data['security_name'] = dt.find_next('span').find_next('span').text.strip()
        elif dt.text.strip() == 'CA Date':
            dates = dt.find_next('dd').select('span')
            data['record_date'] = parse_indonesian_date(dates[0].text.strip())
            data['effective_date'] = parse_indonesian_date(dates[1].text.strip())
            data['start_date'] = dates[2].text.strip()
            data['deadline_date'] = dates[3].text.strip()
        elif dt.text.strip() == 'CA Description':
            data['ca_description'] = dt.find_next('dd').text.strip()

    return data

async def crawl_ksei_data(month: int, year: int, db: Session, event_type: str = None):
    json_data = await fetch_ksei_data(month, year)
    
    color_map = {
        "#565553": "Cum Date",
        "#97191C": "Effective Date",
        "#F27822": "Record Date"
    }
    
    results = []
    
    for event_group in json_data['data']:
        group_event_type = color_map.get(event_group['color'])
        if event_type and group_event_type != event_type:
            continue
        
        for event in event_group['events']:
            detail_url = f"https://www.ksei.co.id{event['description'].replace('\/', '/')}"
            detail_html = await fetch_ksei_detail(detail_url)
            parsed_data = await parse_ksei_detail(detail_html)
            
            # Extract event_date from the URL
            event_date = datetime.strptime(event['description'].split('/')[-1], '%Y-%m-%d')
            
            stock_info = StockInfo(
                event_type=group_event_type,
                event_subtype=parsed_data['event_subtype'],
                stock_name=parsed_data['stock_name'],
                security_code=parsed_data['security_code'],
                security_name=parsed_data['security_name'],
                record_date=parsed_data['record_date'],
                effective_date=parsed_data['effective_date'],
                start_date=parsed_data['start_date'] if parsed_data['start_date'] != '-' else None,
                deadline_date=parsed_data['deadline_date'] if parsed_data['deadline_date'] != '-' else None,
                ca_description=parsed_data['ca_description'],
                crawl_date=datetime.now(),
                event_date=event_date  # Add the new event_date field
            )
            
            db.add(stock_info)
            results.append(stock_info)
    
    db.commit()
    return results

