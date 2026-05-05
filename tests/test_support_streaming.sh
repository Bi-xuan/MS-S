# Helper commands for measuring streaming support-generation memory usage.
/usr/bin/time -l python tests/test_support_streaming.py --mode generator

/usr/bin/time -l python tests/test_support_streaming.py \
  --mode streaming \
  --n 10 \
  --n-edge 10 \
  --limit 100000

/usr/bin/time -l python tests/test_support_streaming.py \
  --mode materialized \
  --n 10 \
  --n-edge 10 \
  --limit 100000

# Look for this line:

# maximum resident set size
# That is peak RAM usage.

# On macOS, it is usually reported in bytes. Convert to MB by dividing by:

# 1024 * 1024
# Example:

# 123456789  maximum resident set size
# means roughly:

# 123456789 / 1024 / 1024 = 117.7 MB
# You should see:

# streaming: peak memory stays relatively low
# materialized: peak memory is much higher and grows as --limit increases
# Try multiple limits:

# /usr/bin/time -l python tests/test_support_streaming.py --mode streaming --n 10 --n-edge 10 --limit 100000
# /usr/bin/time -l python tests/test_support_streaming.py --mode streaming --n 10 --n-edge 10 --limit 500000
# /usr/bin/time -l python tests/test_support_streaming.py --mode streaming --n 10 --n-edge 10 --limit 1000000
# Then compare:

# /usr/bin/time -l python tests/test_support_streaming.py --mode materialized --n 10 --n-edge 10 --limit 100000
# /usr/bin/time -l python tests/test_support_streaming.py --mode materialized --n 10 --n-edge 10 --limit 500000
# /usr/bin/time -l python tests/test_support_streaming.py --mode materialized --n 10 --n-edge 10 --limit 1000000
# Expected pattern:

# streaming:    memory roughly flat
# materialized: memory grows with limit
# You can also use Activity Monitor while it runs, but /usr/bin/time -l gives the cleanest numeric comparison.
