import re
from datetime import datetime, date
from typing import Optional
from jdatetime import datetime as jdatetime, date as jdate


class DateCalculator:
    """A class for handling date calculations in both Gregorian and Jalali calendars."""
    
    JALALI_PREFIXES = ['13', '14']
    GREGORIAN_PREFIXES = ['19', '20']
    
    def __init__(self):
        pass
    
    def validate_date_format(self, date_string: str) -> bool:
        """Validate the date format YYYY-MM or 'present'."""
        if not isinstance(date_string, str):
            return False

        date_string = date_string.strip().lower()
        if date_string == 'present':
            return True
            
        pattern = r'^\d{4}-(0[1-9]|1[0-2])$'
        if not re.match(pattern, date_string):
            return False
        
        return True
    
    def _parse_gregorian_date(self, date_str: str) -> Optional[date]:
        """Parse Gregorian date string to date object."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None
    
    def _parse_jalali_date(self, date_str: str) -> Optional[jdate]:
        """Parse Jalali date string to jdate object."""
        try:
            return jdatetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return None
    
    def _get_current_month_start_gregorian(self) -> date:
        """Get current month start for Gregorian calendar."""
        return date.today().replace(day=1)
    
    def _get_current_month_start_jalali(self) -> jdate:
        """Get current month start for Jalali calendar."""
        return jdate.today().replace(day=1)
    
    def _calculate_month_difference(self, start_date, end_date) -> Optional[int]:
        """Calculate month difference between two dates."""
        if start_date is None or end_date is None:
            return None
        return (end_date - start_date).days // 30
    
    def _get_calendar_type(self, date_str: str) -> Optional[str]:
        """Determine calendar type based on date prefix."""
        if not isinstance(date_str, str) or len(date_str) < 2:
            return None

        prefix = date_str[:2]
        if prefix in self.JALALI_PREFIXES:
            return 'jalali'
        elif prefix in self.GREGORIAN_PREFIXES:
            return 'gregorian'
        return None
    
    def _process_dates(self, start_date_str: str, end_date_str: str, is_jalali: bool) -> Optional[int]:
        """Process dates based on calendar type (Jalali or Gregorian)."""
        # Prepare start date
        full_start_date = f"{start_date_str}-01"
        
        # Parse start date
        start_date = (self._parse_jalali_date if is_jalali else self._parse_gregorian_date)(full_start_date)
        if start_date is None:
            return None
        
        # Prepare end date
        if end_date_str == 'present':
            end_date = (self._get_current_month_start_jalali if is_jalali else self._get_current_month_start_gregorian)()
        else:
            full_end_date = f"{end_date_str}-01"
            end_date = (self._parse_jalali_date if is_jalali else self._parse_gregorian_date)(full_end_date)
            if end_date is None:
                return None
        
        return self._calculate_month_difference(start_date, end_date)
    
    def calculate_duration(self, start_date: str, end_date: str) -> Optional[int]:
        """Calculate duration in months between two dates."""
        try:
            start_date = str(start_date).strip().lower()
            end_date = str(end_date).strip().lower()

            if not self.validate_date_format(start_date):
                return None
            if not self.validate_date_format(end_date):
                return None
            
            start_calendar_type = self._get_calendar_type(start_date)
            
            # For 'present' end date, use start date's calendar type
            if end_date == 'present':
                end_calendar_type = start_calendar_type
            else:
                end_calendar_type = self._get_calendar_type(end_date)
            
            # Validate calendar type compatibility
            if not start_calendar_type or not end_calendar_type:
                return None
            if start_calendar_type != end_calendar_type:
                return None
            
            is_jalali = (start_calendar_type == 'jalali')
            return self._process_dates(start_date, end_date, is_jalali)
        except Exception:
            return None
        

    def calculate_age(self, year: str) -> Optional[int]:
        """Calculate age based on year string."""
        try:
            year_str = str(year).strip()
            if not year_str.isdigit() or len(year_str) < 4:
                return None

            year_prefix = year_str[:2]
            
            if year_prefix in self.JALALI_PREFIXES:
                current_year = jdate.today().year
            elif year_prefix in self.GREGORIAN_PREFIXES:
                current_year = date.today().year
            else:
                return None
            
            return current_year - int(year_str)
        except Exception:
            return None
