#!/usr/bin/env python3
"""
Direct test of the userDataStream endpoint.
"""
import asyncio
import aiohttp
import time
import hmac
import hashlib
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    api_key = os.getenv('BINANCE_API_KEY') or os.getenv('API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET') or os.getenv('API_SECRET')
    
    print(f"Key length: {len(api_key)}, Secret length: {len(api_secret)}")
    
    # Test 1: Direct POST to userDataStream
    async with aiohttp.ClientSession() as session:
        params = {'timestamp': int(time.time() * 1000)}
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        
        url = 'https://api.binance.com/api/v3/userDataStream'
        headers = {'X-MBX-APIKEY': api_key}
        
        print(f'Testing: POST {url}')
        print(f'Params: {list(params.keys())}')
        print(f'Headers: {list(headers.keys())}')
        
        async with session.request('POST', url, params=params, headers=headers) as r:
            text = await r.text()
            print(f'Status: {r.status}')
            print(f'Response:\n{text[:500]}')

if __name__ == '__main__':
    asyncio.run(test())
