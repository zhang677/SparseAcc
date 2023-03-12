Run
```bash
cd spgemm
sqlite3 spgemm.db < spgemm_schema.sql
sqlite3 spgemm.db < load.sql
```
Check
```python
import os
out = os.popen(f'sqlite3 spgemm.db < spgemm.sql').read()
expected = open(f'../data/expected/spgemm.out').read()
assert out == expected
```
