from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .contracts import SessionInfo


NY_TZ = ZoneInfo("America/New_York")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


@dataclass
class SessionCalendar:
    """US-equity-style session calendar with practical half-day rules for MVP."""

    tz: ZoneInfo = NY_TZ

    def is_half_day(self, session_date: date) -> bool:
        if session_date.weekday() >= 5:
            return False

        thanksgiving = _nth_weekday(session_date.year, 11, 3, 4)
        if session_date == thanksgiving + timedelta(days=1):
            return True

        if session_date.month == 12 and session_date.day == 24:
            return True

        if session_date.month == 7 and session_date.day == 3:
            return True

        return False

    def get_session(self, session_date: date) -> SessionInfo:
        open_dt = datetime.combine(session_date, time(9, 30), self.tz)
        close_dt = datetime.combine(
            session_date,
            time(13, 0) if self.is_half_day(session_date) else time(16, 0),
            self.tz,
        )

        entry_start = datetime.combine(session_date, time(10, 0), self.tz)
        default_entry_end = datetime.combine(session_date, time(14, 0), self.tz)
        half_day_entry_end = close_dt - timedelta(hours=1)
        entry_end = min(default_entry_end, half_day_entry_end if self.is_half_day(session_date) else default_entry_end)

        return SessionInfo(
            date=session_date,
            is_half_day=self.is_half_day(session_date),
            open_dt=open_dt,
            close_dt=close_dt,
            entry_start=entry_start,
            entry_end=entry_end,
        )

    def is_in_session(self, ts: datetime) -> bool:
        local_ts = ts.astimezone(self.tz)
        session = self.get_session(local_ts.date())
        return session.open_dt <= local_ts <= session.close_dt

    def is_in_entry_window(self, ts: datetime, session: SessionInfo) -> bool:
        local_ts = ts.astimezone(self.tz)
        return session.entry_start <= local_ts <= session.entry_end
