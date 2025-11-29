import re
from datetime import datetime, date
from jdatetime import datetime as jdatetime, date as jdate


class DateCalculator:
    """A class for handling date calculations in both Gregorian and Jalali calendars."""
    
    JALALI_PREFIXES = ['13', '14']
    GREGORIAN_PREFIXES = ['19', '20']
    
    def __init__(self):
        pass
    
    def validate_date_format(self, date_string: str) -> bool:
        """Validate the date format YYYY-MM or 'present'."""
        if date_string == 'present':
            return True
            
        pattern = r'^\d{4}-(0[1-9]|1[0-2])$'
        if not re.match(pattern, date_string):
            raise ValueError(f"Invalid date format: '{date_string}'. Expected format: YYYY-MM (e.g., 1401-01)")
        
        return True
    
    def _parse_gregorian_date(self, date_str: str) -> date:
        """Parse Gregorian date string to date object."""
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    
    def _parse_jalali_date(self, date_str: str) -> jdate:
        """Parse Jalali date string to jdate object."""
        return jdatetime.strptime(date_str, "%Y-%m-%d").date()
    
    def _get_current_month_start_gregorian(self) -> date:
        """Get current month start for Gregorian calendar."""
        return date.today().replace(day=1)
    
    def _get_current_month_start_jalali(self) -> jdate:
        """Get current month start for Jalali calendar."""
        return jdate.today().replace(day=1)
    
    def _calculate_month_difference(self, start_date, end_date) -> int:
        """Calculate month difference between two dates."""
        return (end_date - start_date).days // 30
    
    def _get_calendar_type(self, date_str: str) -> str:
        """Determine calendar type based on date prefix."""
        prefix = date_str[:2]
        if prefix in self.JALALI_PREFIXES:
            return 'jalali'
        elif prefix in self.GREGORIAN_PREFIXES:
            return 'gregorian'
        else:
            raise ValueError(f"Unsupported date prefix: {prefix}")
    
    def _process_dates(self, start_date_str: str, end_date_str: str, is_jalali: bool) -> int:
        """Process dates based on calendar type (Jalali or Gregorian)."""
        # Prepare start date
        full_start_date = f"{start_date_str}-01"
        
        # Parse start date
        start_date = (self._parse_jalali_date if is_jalali else self._parse_gregorian_date)(full_start_date)
        
        # Prepare end date
        if end_date_str == 'present':
            end_date = (self._get_current_month_start_jalali if is_jalali else self._get_current_month_start_gregorian)()
        else:
            full_end_date = f"{end_date_str}-01"
            end_date = (self._parse_jalali_date if is_jalali else self._parse_gregorian_date)(full_end_date)
        
        return self._calculate_month_difference(start_date, end_date)
    
    def calculate_duration(self, start_date: str, end_date: str) -> int:
        """Calculate duration in months between two dates."""
        self.validate_date_format(start_date)
        self.validate_date_format(end_date)
        
        start_calendar_type = self._get_calendar_type(start_date)
        
        # For 'present' end date, use start date's calendar type
        if end_date == 'present':
            end_calendar_type = start_calendar_type
        else:
            end_calendar_type = self._get_calendar_type(end_date)
        
        # Validate calendar type compatibility
        if start_calendar_type != end_calendar_type:
            raise ValueError("Incompatible calendar types between start and end dates")
        
        is_jalali = (start_calendar_type == 'jalali')
        return self._process_dates(start_date, end_date, is_jalali)
    
    def calculate_age(self, year: str) -> int:
        """Calculate age based on year string."""
        year_prefix = str(year)[:2]
        
        if year_prefix in self.JALALI_PREFIXES:
            current_year = jdate.today().year
        elif year_prefix in self.GREGORIAN_PREFIXES:
            current_year = date.today().year
        else:
            raise ValueError(f"Unsupported year prefix: {year_prefix}")
        
        return current_year - int(year)