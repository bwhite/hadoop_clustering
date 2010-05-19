#!/usr/bin/env python
import hadoopy
import time
import json


class ProfileJob(object):
    def __init__(self):
        self._times = {}
        self._pending_times = {}
        self.start_time('total')
        
    def start_time(self, name):
        self._pending_times[name] = time.time()

    def stop_time(self, name):
        try:
            dur = time.time() - self._pending_times[name]
        except KeyError:
            hadoopy.counter('stop_time', 'timer_failed')
        else:
            try:
                time_stats = self._times[name]
                # Min/Max/Sum/Count
                self._times[name] = [min(time_stats[0], dur),
                                     max(time_stats[1], dur),
                                     time_stats[2] + dur,
                                     time_stats[3] + 1]
            except KeyError:
                self._times[name] = [dur, dur, dur, 1]

    def close(self):
        self.stop_time('total')
        hadoopy.status(json.dumps(self._times))
