#!/usr/bin/env python3

try:
    from logic.info_strings import *
    print('✅ info_strings.py imported successfully - no syntax errors!')
except Exception as e:
    print(f'❌ Error importing info_strings.py: {e}')
    import traceback
    traceback.print_exc() 