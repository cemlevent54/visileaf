"""
Email service - Email sending with Resend API
"""
import os
import logging
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import resend

# Initialize logger
logger = logging.getLogger(__name__)

# Resend API configuration
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = "onboarding@resend.dev"

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class EmailService:
    """Email service for sending emails via Resend API"""
    
    def __init__(self):
        """
        Initialize email service with Resend configuration
        """
        # Set Resend API key
        if RESEND_API_KEY:
            resend.api_key = RESEND_API_KEY
            logger.info('EmailService: Resend API key configured', extra={
                'has_api_key': True,
                'from_email': RESEND_FROM_EMAIL
            })
        else:
            logger.warning('EmailService: RESEND_API_KEY not found in environment variables')
    
    def _get_default_from(self) -> str:
        """
        Get default from email address
        
        Returns:
            Default from email address
        """
        return RESEND_FROM_EMAIL
    
    def send_mail_with_resend(
        self,
        to: Union[str, List[str]],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send email using Resend API
        
        Args:
            to: Recipient email address(es) - string or list of strings
            options: Email options dictionary containing:
                - subject: Email subject (required)
                - html: HTML content (optional)
                - text: Plain text content (optional, used if html not provided)
                - cc: CC recipients (optional) - string or list
                - bcc: BCC recipients (optional) - string or list
                - from: From email address (optional, uses default if not provided)
                - tags: Email tags for tracking (optional) - list of dicts
        
        Returns:
            Dictionary with 'success' boolean and 'data' or 'error' keys
            Example: {'success': True, 'data': {...}} or {'success': False, 'error': {...}}
        """
        if options is None:
            options = {}
        
        try:
            # Extract options
            subject = options.get('subject', '')
            html = options.get('html')
            text = options.get('text')
            cc = options.get('cc')
            bcc = options.get('bcc')
            from_email = options.get('from')
            tags = options.get('tags')
            attachments = options.get('attachments')
            
            # Prepare recipient list
            to_list = [to] if isinstance(to, str) else to
            
            # Prepare email options
            mail_options: Dict[str, Any] = {
                'from': from_email or self._get_default_from(),
                'to': to_list,
                'subject': subject,
            }
            
            # Add content (html takes priority over text)
            if html:
                mail_options['html'] = html
            elif text:
                mail_options['text'] = text
            
            # Add optional fields
            if cc:
                mail_options['cc'] = [cc] if isinstance(cc, str) else cc
            
            if bcc:
                mail_options['bcc'] = [bcc] if isinstance(bcc, str) else bcc
            
            if attachments:
                mail_options['attachments'] = attachments
            
            if tags:
                mail_options['tags'] = tags
            
            # Log request (redact sensitive info)
            logger.debug('EmailService: send_mail_with_resend request', extra={
                'to': mail_options.get('to'),
                'subject': mail_options.get('subject'),
                'has_html': 'html' in mail_options,
                'has_text': 'text' in mail_options,
                'cc': mail_options.get('cc'),
                'bcc': '[REDACTED]' if mail_options.get('bcc') else None,
                'from': mail_options.get('from'),
                'tags': mail_options.get('tags'),
            })
            
            # Send email via Resend
            data = resend.Emails.send(mail_options)
            
            # Log success
            logger.info('EmailService: mail sent with Resend', extra={
                'to': mail_options.get('to'),
                'subject': mail_options.get('subject'),
                'email_id': data.get('id') if isinstance(data, dict) else None,
            })
            
            return {
                'success': True,
                'data': data
            }
            
        except Exception as err:
            # Log error
            error_info = {
                'error': str(err),
                'error_type': type(err).__name__,
            }
            
            # Add additional error info if available
            if hasattr(err, 'status_code'):
                error_info['status_code'] = err.status_code
            if hasattr(err, 'message'):
                error_info['error_message'] = err.message  # Changed from 'message' to 'error_message' to avoid LogRecord conflict
            
            logger.error('EmailService: send_mail_with_resend error', extra=error_info, exc_info=True)
            
            return {
                'success': False,
                'error': error_info
            }
    
    def _render_template(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Render HTML template with variables
        
        Args:
            template_name: Template file name (e.g., 'forgot_password.html')
            variables: Dictionary of variables to replace in template
        
        Returns:
            Rendered HTML string
        
        Raises:
            FileNotFoundError: If template file not found
            Exception: If template rendering fails
        """
        template_path = TEMPLATES_DIR / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple template variable replacement: {{variable}} -> value
            import re
            for key, value in variables.items():
                # Replace {{key}} with value using regex
                pattern = re.compile(r'\{\{' + re.escape(key) + r'\}\}')
                content = pattern.sub(str(value or ''), content)
                
            # Clean up any remaining {{...}} patterns (if not found in variables)
            content = re.sub(r'\{\{[^}]+\}\}', '', content)
            
            return content
            
        except Exception as e:
            logger.error(f'EmailService: template rendering error', extra={
                'template': template_name,
                'error': str(e)
            }, exc_info=True)
            raise

