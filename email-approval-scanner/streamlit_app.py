import streamlit as st
import imaplib
import email
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import pickle
import os
import urllib.parse
import json

# Set page config ONCE at the very top
st.set_page_config(
    page_title="STRICT AI Email Approval Scanner",
    page_icon="🤖",
    layout="wide"
)

# Machine Learning imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError as e:
    st.error(f"ML libraries missing: {e}")
    st.stop()

# Gmail API imports for real message IDs
GMAIL_API_AVAILABLE = False
try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    import base64
    GMAIL_API_AVAILABLE = True
    print("✅ Gmail API libraries available - Direct email links enabled")
except ImportError:
    GMAIL_API_AVAILABLE = False
    print("❌ Gmail API libraries not available - Using IMAP fallback")

# Credential management functions
def save_credentials_to_session(email, password):
    """Save credentials to streamlit session with persistence"""
    st.session_state.persistent_email = email
    st.session_state.persistent_password = password
    st.session_state.credentials_saved = True
    
    credentials = {
        'email': email,
        'password': password,
        'saved_at': datetime.now().isoformat()
    }
    
    try:
        with open('.streamlit_credentials.json', 'w') as f:
            json.dump(credentials, f)
    except:
        pass

def load_persistent_credentials():
    """Load credentials from persistent storage"""
    try:
        if os.path.exists('.streamlit_credentials.json'):
            with open('.streamlit_credentials.json', 'r') as f:
                credentials = json.load(f)
                return credentials.get('email'), credentials.get('password')
    except:
        pass
    return None, None

class StrictApprovalAIModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        self.training_emails = [
            ("With reference to the below mail, all GADs and Data sheets are ok. Kindly submit the QAP for the same.", 1),
            ("Dear Sir, Approved drawing is enclosed, Kindly arrange to share the QAP also.", 1),
            ("Dear sir, These are the approval of documents for single girder EOT crane. Please review this and start with manufacturing.", 1),
            ("Please find attached Approved Drg & kindly start Manufacturing.", 1),
            ("Please find enclosed herewith Approved Datasheet with GA Drawing. Kindly proceed for dispatch", 1),
            ("We have gone through the GAD and it is approved from our end.", 1),
            ("Dear Team, PFA approved Cat-I document. Kindly start the manufacturing and share your manufacturing & inspection schedule.", 1),
            ("Please acknowledge the same Drawing and QAP is approved, please proceed for manufacturing.", 1),
            ("PFA the GA & data sheet approved. Thanks & Regards", 1),
            ("We hereby approve the attached drawing. You are requested to supply as per the revised GA", 1),
            ("Please get the drawing approved from your side and submit for our review", 0),
            ("Do the needful for drg approval from your technical team", 0),
            ("We require approval from you before we can proceed with this project", 0),
            ("Please revise the drawing as per comments and resubmit for approval", 0),
            ("Kindly get internal approval and then submit the documents", 0),
            ("Please submit the drawing for our approval", 0),
            ("Drawing approval is required from your end", 0),
            ("We are waiting for your approval on the submitted documents", 0)
        ]
    
    def preprocess_text(self, text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'from:.*?to:.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'sent:.*?\n', '', text)
        text = re.sub(r'subject:.*?\n', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\+?[\d\s\-\(\)]{10,}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_strict_features(self, text):
        text_lower = text.lower()
        
        positive_features = {
            'has_approved_document': int(bool(re.search(r'approved.{0,20}(drawing|document|ga|datasheet|drg)', text_lower))),
            'has_document_approved': int(bool(re.search(r'(drawing|document|ga|datasheet|drg).{0,20}approved', text_lower))),
            'has_approved_attached': int('approved' in text_lower and ('attached' in text_lower or 'enclosed' in text_lower)),
            'has_pfa_approved': int('pfa' in text_lower and 'approved' in text_lower),
            'has_find_approved': int('find' in text_lower and 'approved' in text_lower),
            'has_start_manufacturing': int('start' in text_lower and 'manufacturing' in text_lower),
            'has_proceed_manufacturing': int('proceed' in text_lower and 'manufacturing' in text_lower),
            'has_go_ahead': int('go ahead' in text_lower),
            'has_formal_approval': int('formal approval' in text_lower),
            'has_hereby_approve': int('hereby approve' in text_lower),
            'has_approved_from_end': int('approved from our end' in text_lower),
            'has_mfc': int('mfc' in text_lower),
            'has_signed_stamped': int('signed' in text_lower and 'stamp' in text_lower),
        }
        
        negative_features = {
            'requests_approval': int(bool(re.search(r'(please|kindly).{0,30}(get|provide|arrange).{0,20}approval', text_lower))),
            'needs_approval': int(bool(re.search(r'(need|require|want).{0,20}approval', text_lower))),
            'approval_from_you': int('approval from you' in text_lower or 'your approval' in text_lower),
            'get_approved': int('get' in text_lower and 'approved' in text_lower and 'from' in text_lower),
            'submit_for_approval': int('submit' in text_lower and 'approval' in text_lower),
            'approval_required': int('approval required' in text_lower),
            'approval_pending': int('approval pending' in text_lower),
        }
        
        all_features = {**positive_features, **negative_features}
        all_features.update({
            'word_count': len(text.split()),
            'has_question_mark': int('?' in text),
            'has_please': int('please' in text_lower),
        })
        
        return all_features
    
    def train_model(self):
        texts = []
        labels = []
        
        for text, label in self.training_emails:
            processed_text = self.preprocess_text(text)
            texts.append(processed_text)
            labels.append(label)
        
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_text = self.vectorizer.fit_transform(texts)
        
        additional_features = []
        for text in texts:
            features = self.extract_strict_features(text)
            additional_features.append(list(features.values()))
        
        X_additional = np.array(additional_features)
        X_combined = np.hstack([X_text.toarray(), X_additional])
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight={0: 1, 1: 2}
        )
        
        self.model.fit(X_combined, labels)
        self.is_trained = True
        return accuracy_score(labels, self.model.predict(X_combined))
    
    def predict_approval(self, email_text, subject="", sender=""):
        if not self.is_trained:
            self.train_model()
        
        full_text = f"{subject} {email_text}"
        processed_text = self.preprocess_text(full_text)
        
        # Ultra strict rejection patterns
        ultra_strict_rejection_patterns = [
            r'please.{0,50}get.{0,20}approval',
            r'kindly.{0,50}get.{0,20}approval',
            r'need.{0,30}approval.{0,30}from',
            r'require.{0,30}approval.{0,30}from',
            r'approval.{0,30}from.{0,30}you',
            r'submit.{0,30}for.{0,20}approval',
            r'approval.{0,20}required',
            r'approval.{0,20}pending',
        ]
        
        for pattern in ultra_strict_rejection_patterns:
            if re.search(pattern, processed_text):
                return {
                    'is_approved': False,
                    'confidence': 98.0,
                    'probability_approved': 2.0,
                    'reasoning': [f'Ultra strict rejection: {pattern[:30]}'],
                    'features_detected': ['ultra_strict_rejection_pattern']
                }
        
        # Must have explicit approval language
        must_have_approval_patterns = [
            r'approved.{0,20}(drawing|document|ga|datasheet|drg)',
            r'(drawing|document|ga|datasheet|drg).{0,20}approved',
            r'hereby\s+approve',
            r'formal\s+approval',
            r'approved\s+from\s+our\s+end',
            r'pfa.{0,20}approved',
        ]
        
        has_explicit_approval = any(re.search(pattern, processed_text) for pattern in must_have_approval_patterns)
        
        if not has_explicit_approval:
            return {
                'is_approved': False,
                'confidence': 85.0,
                'probability_approved': 15.0,
                'reasoning': ['No explicit approval language'],
                'features_detected': ['no_explicit_approval']
            }
        
        # ML prediction
        X_text = self.vectorizer.transform([processed_text])
        features = self.extract_strict_features(processed_text)
        X_additional = np.array([list(features.values())])
        X_combined = np.hstack([X_text.toarray(), X_additional])
        
        probabilities = self.model.predict_proba(X_combined)[0]
        approval_prob = probabilities[1] * 100
        
        ULTRA_STRICT_THRESHOLD = 0.85
        is_approved = probabilities[1] > ULTRA_STRICT_THRESHOLD
        
        if is_approved:
            strong_positives = [
                features['has_approved_document'],
                features['has_document_approved'],
                features['has_start_manufacturing'],
                features['has_proceed_manufacturing'],
                features['has_go_ahead'],
                features['has_formal_approval'],
                features['has_hereby_approve'],
                features['has_approved_attached'],
                features['has_pfa_approved'],
                features['has_find_approved']
            ]
            
            if sum(strong_positives) < 2:
                is_approved = False
                approval_prob = min(approval_prob, 35.0)
        
        confidence = max(probabilities) * 100
        
        return {
            'is_approved': is_approved,
            'confidence': confidence,
            'probability_approved': approval_prob,
            'reasoning': [f"ML confidence: {confidence:.1f}%"],
            'features_detected': [k for k, v in features.items() if v > 0]
        }

class GmailAPIScanner:
    """Gmail API scanner that provides real Gmail message IDs like the working tkinter code"""
    
    def __init__(self, email, password):
        self.email = email
        self.password = password
        self.service = None
        self.ai_model = StrictApprovalAIModel()
        self.scopes = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def authenticate_gmail_api(self):
        """Authenticate with Gmail API"""
        try:
            creds = None
            
            # The file token.json stores the user's access and refresh tokens
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', self.scopes)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists('credentials.json'):
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'credentials.json', self.scopes)
                        creds = flow.run_local_server(port=0)
                    else:
                        st.error("Gmail API credentials.json not found. Using IMAP fallback.")
                        return False
                
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('gmail', 'v1', credentials=creds)
            return True
            
        except Exception as e:
            st.warning(f"Gmail API authentication failed: {e}")
            return False
    
    def scan_recent_emails_api(self, days=7, max_emails=None, company_domains=None):
        """Scan emails using Gmail API - provides real Gmail message IDs"""
        if company_domains is None:
            company_domains = ['revacranes.com', 'eiepl.in']
            
        try:
            if not self.service:
                if not self.authenticate_gmail_api():
                    return [], []
            
            # Calculate date range
            since_date = datetime.now() - timedelta(days=days)
            query = f'after:{since_date.strftime("%Y/%m/%d")}'
            
            # Get messages
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_emails
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return [], []
            
            approved_emails = []
            all_analyzed = []
            
            total_messages = len(messages)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, message in enumerate(messages):
                progress_bar.progress((i + 1) / total_messages)
                status_text.text(f"Gmail API analyzing email {i+1}/{total_messages}")
                
                try:
                    # Get Gmail message ID (REAL Gmail message ID!)
                    gmail_msg_id = message['id']
                    
                    msg_detail = self.service.users().messages().get(
                        userId='me', id=gmail_msg_id
                    ).execute()
                    
                    # Extract headers
                    headers = {h['name']: h['value'] for h in msg_detail['payload'].get('headers', [])}
                    
                    subject = headers.get('Subject', '(No Subject)')
                    sender = headers.get('From', 'Unknown Sender')
                    date = headers.get('Date', 'Unknown Date')
                    
                    # Skip internal emails
                    sender_domain = sender.split('@')[-1].lower() if '@' in sender else ""
                    if any(domain in sender_domain for domain in company_domains):
                        continue
                    
                    # Get email content
                    content = self.get_email_content_api(msg_detail)
                    
                    if not content:
                        continue
                    
                    # AI Analysis
                    ai_result = self.ai_model.predict_approval(content, subject, sender)
                    
                    email_data = {
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'content_preview': content[:500] + "..." if len(content) > 500 else content,
                        'is_approved': ai_result['is_approved'],
                        'confidence': ai_result['confidence'],
                        'probability_approved': ai_result['probability_approved'],
                        'ai_reasoning': ai_result['reasoning'],
                        'features_detected': ai_result['features_detected'],
                        'gmail_message_id': gmail_msg_id,  # REAL Gmail message ID!
                        'email_id': gmail_msg_id
                    }
                    
                    all_analyzed.append(email_data)
                    
                    if ai_result['is_approved']:
                        approved_emails.append(email_data)
                
                except Exception as e:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            return approved_emails, all_analyzed
            
        except Exception as e:
            st.error(f"Gmail API error: {e}")
            return [], []
    
    def get_email_content_api(self, msg_detail):
        """Extract email content from Gmail API response"""
        try:
            payload = msg_detail['payload']
            
            def extract_text(payload):
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            data = part['body']['data']
                            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        elif 'parts' in part:
                            return extract_text(part)
                elif payload['mimeType'] == 'text/plain':
                    data = payload['body']['data']
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                return ""
            
            content = extract_text(payload)
            return self.extract_latest_email_only(content)
            
        except Exception as e:
            return ""
    
    def extract_latest_email_only(self, email_content):
        """Extract only the latest email content, removing forwarded/replied content"""
        if not email_content:
            return ""
        
        reply_patterns = [
            r'-----Original Message-----',
            r'From:.*?Sent:.*?To:.*?Subject:',
            r'On.*?wrote:',
            r'> ', r'>>',
            r'From: .*?@.*?\n',
            r'________________________________',
        ]
        
        lines = email_content.split('\n')
        latest_content_lines = []
        
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in reply_patterns):
                break
            if not latest_content_lines and not line.strip():
                continue
            latest_content_lines.append(line)
        
        return '\n'.join(latest_content_lines).strip()
    
    def close_connection(self):
        """Gmail API doesn't need explicit connection closing"""
        pass

class EmailScanner:
    def __init__(self, email, password, imap_server="imap.gmail.com"):
        self.email = email
        self.password = password
        self.imap_server = imap_server
        self.mail = None
        self.ai_model = StrictApprovalAIModel()
        self.is_connected = False
    
    def connect_to_email(self):
        try:
            if self.is_connected and self.mail:
                try:
                    self.mail.noop()
                    return True
                except:
                    self.is_connected = False
                    self.mail = None
            
            st.info("Connecting to Gmail IMAP server...")
            self.mail = imaplib.IMAP4_SSL(self.imap_server, 993)
            
            try:
                login_result = self.mail.login(self.email, self.password)
                if login_result[0] == 'OK':
                    st.success("Successfully connected to Gmail!")
                    self.mail.select("inbox")
                    self.is_connected = True
                    return True
                else:
                    st.error(f"Login failed: {login_result}")
                    return False
            except imaplib.IMAP4.error as e:
                error_msg = str(e).lower()
                if "invalid credentials" in error_msg or "authentication failed" in error_msg:
                    st.error("""
                    Authentication Failed! 
                    
                    Generate a NEW Gmail App Password:
                    1. Go to Google Account > Security
                    2. Enable 2-Step Verification
                    3. Go to App Passwords
                    4. Generate new password for "Mail"
                    5. Use the 16-character code
                    """)
                else:
                    st.error(f"IMAP Error: {e}")
                return False
            
        except Exception as e:
            st.error(f"Connection error: {e}")
            return False
    
    def get_email_content(self, msg):
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        charset = part.get_content_charset() or 'utf-8'
                        body = part.get_payload(decode=True).decode(charset, errors='ignore')
                        break
                    except:
                        continue
        else:
            try:
                charset = msg.get_content_charset() or 'utf-8'
                body = msg.get_payload(decode=True).decode(charset, errors='ignore')
            except:
                body = str(msg.get_payload())
        
        return self.extract_latest_email_only(body)
    
    def extract_latest_email_only(self, email_content):
        if not email_content:
            return ""
        
        reply_patterns = [
            r'-----Original Message-----',
            r'From:.*?Sent:.*?To:.*?Subject:',
            r'On.*?wrote:',
            r'> ', r'>>',
            r'From: .*?@.*?\n',
            r'________________________________',
        ]
        
        lines = email_content.split('\n')
        latest_content_lines = []
        
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in reply_patterns):
                break
            if not latest_content_lines and not line.strip():
                continue
            latest_content_lines.append(line)
        
        return '\n'.join(latest_content_lines).strip()
    
    def is_external_email(self, sender, company_domains):
        sender_domain = sender.split('@')[-1].lower() if '@' in sender else ""
        return not any(domain in sender_domain for domain in company_domains)
    
    def get_gmail_message_id(self, msg, email_id):
        """Extract Gmail message ID from email headers"""
        try:
            # Try to get X-GM-MSGID header (Gmail specific)
            gmail_msgid = msg.get('X-GM-MSGID')
            if gmail_msgid:
                return gmail_msgid
            
            # Try to get Message-ID and convert to Gmail format
            message_id = msg.get('Message-ID', '')
            if message_id:
                # Remove angle brackets and convert to hex
                clean_id = message_id.strip('<>')
                # Try to extract a usable part for Gmail
                if '@' in clean_id:
                    local_part = clean_id.split('@')[0]
                    # Convert to a format Gmail might recognize
                    try:
                        return format(hash(local_part) & 0x7FFFFFFFFFFFFFFF, 'x')
                    except:
                        pass
            
            # Fallback: use IMAP UID converted to hex
            try:
                if isinstance(email_id, bytes):
                    email_id = email_id.decode()
                return format(int(email_id), 'x')
            except:
                return email_id
                
        except Exception:
            return str(email_id)
    
    def scan_recent_emails(self, days=7, max_emails=None, company_domains=None):
        if company_domains is None:
            company_domains = ['revacranes.com', 'eiepl.in']
            
        if not self.mail:
            if not self.connect_to_email():
                return [], []
        
        try:
            since_date = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
            result, data = self.mail.search(None, f'(SINCE "{since_date}")')
            
            if result != 'OK':
                return [], []
            
            email_ids = data[0].split()
            
            if max_emails and max_emails > 0 and len(email_ids) > max_emails:
                email_ids = email_ids[-max_emails:]
            
            approved_emails = []
            all_analyzed = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, email_id in enumerate(email_ids):
                progress_bar.progress((i + 1) / len(email_ids))
                status_text.text(f"Analyzing email {i+1}/{len(email_ids)}")
                
                try:
                    # Fetch email with headers to get Gmail message ID
                    result, msg_data = self.mail.fetch(email_id, "(RFC822)")
                    if result != 'OK':
                        continue
                    
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    subject = msg.get("subject", "(No Subject)")
                    sender = msg.get("from", "Unknown Sender")
                    date = msg.get("date", "Unknown Date")
                    
                    if not self.is_external_email(sender, company_domains):
                        continue
                    
                    content = self.get_email_content(msg)
                    if not content:
                        continue
                    
                    # Get Gmail message ID
                    gmail_message_id = self.get_gmail_message_id(msg, email_id)
                    
                    ai_result = self.ai_model.predict_approval(content, subject, sender)
                    
                    email_data = {
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'content_preview': content[:500] + "..." if len(content) > 500 else content,
                        'is_approved': ai_result['is_approved'],
                        'confidence': ai_result['confidence'],
                        'probability_approved': ai_result['probability_approved'],
                        'ai_reasoning': ai_result['reasoning'],
                        'features_detected': ai_result['features_detected'],
                        'email_id': email_id.decode() if isinstance(email_id, bytes) else str(email_id),
                        'gmail_message_id': gmail_message_id
                    }
                    
                    all_analyzed.append(email_data)
                    
                    if ai_result['is_approved']:
                        approved_emails.append(email_data)
                
                except Exception as e:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            return approved_emails, all_analyzed
            
        except Exception as e:
            st.error(f"Error scanning emails: {e}")
            return [], []
    
    def close_connection(self):
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
            except:
                pass

def generate_gmail_link(email_data, gmail_account_index, user_email):
    """Generate Gmail link using the proper message ID"""
    try:
        # Use the Gmail message ID we extracted
        message_id = email_data.get('gmail_message_id', '')
        
        if not message_id:
            # Fallback to email_id if gmail_message_id is not available
            email_id = email_data.get('email_id', '')
            if email_id.isdigit():
                message_id = format(int(email_id), 'x')
            else:
                message_id = email_id
        
        # Use the exact format that was working before
        gmail_url = f"https://mail.google.com/mail/u/{gmail_account_index}/?authuser={user_email}#all/{message_id}"
        return gmail_url
        
    except Exception as e:
        # Fallback to inbox if anything goes wrong
        return f"https://mail.google.com/mail/u/{gmail_account_index}/?authuser={user_email}#inbox"

def main():
    st.title("🤖 STRICT AI Email Approval Scanner")
    st.caption("AI-powered crane approval detection - Now available globally via web!")
    
    # Initialize session state
    if 'credentials_saved' not in st.session_state:
        saved_email, saved_password = load_persistent_credentials()
        if saved_email and saved_password:
            st.session_state.persistent_email = saved_email
            st.session_state.persistent_password = saved_password
            st.session_state.credentials_saved = True
        else:
            st.session_state.credentials_saved = False
    
    if 'gmail_account_index' not in st.session_state:
        st.session_state.gmail_account_index = 2
    
    if 'use_gmail_api' not in st.session_state:
        st.session_state.use_gmail_api = GMAIL_API_AVAILABLE
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Email Settings")
        
        if not st.session_state.credentials_saved:
            st.warning("Enter credentials once - they'll be remembered!")
            user_email = st.text_input("Your Email:", value="design@revacranes.com")
            user_password = st.text_input("Gmail App Password:", type="password")
            
            if st.button("💾 Save & Connect", type="primary"):
                if user_email and user_password:
                    # Try Gmail API first if available
                    if GMAIL_API_AVAILABLE:
                        test_scanner = GmailAPIScanner(user_email, user_password)
                        if test_scanner.authenticate_gmail_api():
                            save_credentials_to_session(user_email, user_password)
                            st.session_state.use_gmail_api = True
                            st.success("Gmail API connected! (Real message IDs)")
                            st.rerun()
                        else:
                            # Fallback to IMAP
                            test_scanner = EmailScanner(user_email, user_password)
                            if test_scanner.connect_to_email():
                                save_credentials_to_session(user_email, user_password)
                                st.session_state.use_gmail_api = False
                                test_scanner.close_connection()
                                st.success("IMAP connected!")
                                st.rerun()
                            else:
                                st.error("Both Gmail API and IMAP failed.")
                    else:
                        # IMAP only
                        test_scanner = EmailScanner(user_email, user_password)
                        if test_scanner.connect_to_email():
                            save_credentials_to_session(user_email, user_password)
                            test_scanner.close_connection()
                            st.success("IMAP connected!")
                            st.rerun()
                        else:
                            st.error("Connection failed.")
                else:
                    st.error("Please enter both email and password")
        else:
            connection_type = "Gmail API" if st.session_state.get('use_gmail_api') else "IMAP"
            st.success(f"✅ Connected: {st.session_state.persistent_email}")
            st.info(f"Using: {connection_type}")
            if st.button("🔄 Change Credentials"):
                st.session_state.credentials_saved = False
                try:
                    if os.path.exists('.streamlit_credentials.json'):
                        os.remove('.streamlit_credentials.json')
                except:
                    pass
                st.rerun()
        
        gmail_account_index = st.selectbox(
            "Gmail Account Position:",
            options=[0, 1, 2, 3, 4],
            index=st.session_state.gmail_account_index
        )
        st.session_state.gmail_account_index = gmail_account_index
        
        company_domains = st.text_area(
            "Company Domains:",
            value="revacranes.com\neiepl.in"
        ).strip().split('\n')
        
        st.subheader("Scan Settings")
        days_to_scan = st.selectbox("Scan Last:", [1, 3, 7, 14, 30], index=2)
        max_emails = st.selectbox("Email Limit:", ["ALL", 50, 100, 200], index=0)
    
    # Main interface
    if not st.session_state.credentials_saved:
        st.info("👈 Please save your email credentials in the sidebar")
        return
    
    # Scan button
    if st.button("🎯 Start STRICT AI Scan", type="primary"):
        # Choose scanner based on what was successful during connection
        if st.session_state.get('use_gmail_api') and GMAIL_API_AVAILABLE:
            scanner = GmailAPIScanner(
                st.session_state.persistent_email, 
                st.session_state.persistent_password
            )
            st.info("Using Gmail API for real message IDs...")
        else:
            scanner = EmailScanner(
                st.session_state.persistent_email, 
                st.session_state.persistent_password
            )
            st.info("Using IMAP scanner...")
        
        max_emails_value = None if max_emails == "ALL" else max_emails
        
        with st.spinner("STRICT AI analyzing emails..."):
            if st.session_state.get('use_gmail_api') and hasattr(scanner, 'scan_recent_emails_api'):
                approved_emails, all_emails = scanner.scan_recent_emails_api(
                    days=days_to_scan,
                    max_emails=max_emails_value,
                    company_domains=company_domains
                )
            else:
                approved_emails, all_emails = scanner.scan_recent_emails(
                    days=days_to_scan,
                    max_emails=max_emails_value,
                    company_domains=company_domains
                )
        
        scanner.close_connection()
        
        st.session_state.approved_emails = approved_emails
        st.session_state.all_emails = all_emails
        
        scanner_type = "Gmail API" if st.session_state.get('use_gmail_api') else "IMAP"
        st.success(f"Found {len(approved_emails)} approvals out of {len(all_emails)} emails using {scanner_type}.")
    
    # Display results
    if 'approved_emails' in st.session_state:
        tab1, tab2 = st.tabs(["✅ Approvals", "📧 All Emails"])
        
        with tab1:
            st.subheader(f"High Confidence Approvals ({len(st.session_state.approved_emails)})")
            
            for i, email in enumerate(st.session_state.approved_emails):
                with st.expander(f"📧 {email['subject'][:60]}... | {email['confidence']:.1f}%"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**From:** {email['sender']}")
                        st.write(f"**Date:** {email['date']}")
                        st.write(f"**Confidence:** {email['confidence']:.1f}%")
                        st.text_area("Content:", email['content_preview'], height=100, key=f"content_{i}")
                    
                    with col2:
                        gmail_link = generate_gmail_link(email, st.session_state.gmail_account_index, st.session_state.persistent_email)
                        
                        # Direct clickable link that opens in new tab
                        st.markdown(f"""
                        <a href="{gmail_link}" target="_blank" style="
                            display: inline-block;
                            background-color: #1f77b4;
                            color: white;
                            padding: 10px 20px;
                            text-decoration: none;
                            border-radius: 5px;
                            margin: 5px 0;
                        ">📧 Open Gmail Directly</a>
                        """, unsafe_allow_html=True)
                        
                        # Show copy link button instead of nested expander
                        if st.button("📋 Copy Link", key=f"copy_{i}"):
                            st.code(gmail_link, language=None)
                            st.success("Link displayed above - copy it!")
                        
                        if st.button("🤖 AI Analysis", key=f"ai_{i}"):
                            st.json({
                                'is_approved': email['is_approved'],
                                'confidence': f"{email['confidence']:.1f}%",
                                'reasoning': email['ai_reasoning'],
                                'features': email['features_detected']
                            })
        
        with tab2:
            st.subheader(f"All Analyzed Emails ({len(st.session_state.all_emails)})")
            
            if st.session_state.all_emails:
                # Show all emails in expandable format
                for i, email in enumerate(st.session_state.all_emails):
                    # Determine status
                    status = "✅ APPROVED" if email['is_approved'] else "❌ Not Approved"
                    confidence = email['confidence']
                    
                    with st.expander(f"{status} | {email['subject'][:50]}... | {confidence:.1f}%"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**From:** {email['sender']}")
                            st.write(f"**Date:** {email['date']}")
                            st.write(f"**AI Decision:** {status}")
                            st.write(f"**Confidence:** {confidence:.1f}%")
                            st.text_area("Content:", email['content_preview'], height=80, key=f"all_content_{i}")
                        
                        with col2:
                            gmail_link = generate_gmail_link(email, st.session_state.gmail_account_index, st.session_state.persistent_email)
                            
                            # Direct clickable link
                            st.markdown(f"""
                            <a href="{gmail_link}" target="_blank" style="
                                display: inline-block;
                                background-color: #1f77b4;
                                color: white;
                                padding: 8px 16px;
                                text-decoration: none;
                                border-radius: 4px;
                                margin: 4px 0;
                            ">📧 Open Gmail</a>
                            """, unsafe_allow_html=True)
                            
                            if st.button("📋 Copy Link", key=f"copy_all_{i}"):
                                st.code(gmail_link, language=None)
                                st.success("Link displayed!")
                            
                            if st.button("🤖 AI Details", key=f"ai_all_{i}"):
                                st.json({
                                    'is_approved': email['is_approved'],
                                    'confidence': f"{confidence:.1f}%",
                                    'reasoning': email['ai_reasoning'],
                                    'features': email['features_detected']
                                })
                
                # Summary table for overview
                st.subheader("Summary Table")
                df_data = []
                for email in st.session_state.all_emails:
                    sender_short = email['sender'].split('<')[-1].split('>')[0] if '<' in email['sender'] else email['sender'][:30]
                    df_data.append({
                        'Subject': email['subject'][:40] + "..." if len(email['subject']) > 40 else email['subject'],
                        'From': sender_short,
                        'Decision': "✅ APPROVED" if email['is_approved'] else "❌ Not Approved",
                        'Confidence': f"{email['confidence']:.1f}%",
                        'Date': email['date'][:10] if len(email['date']) > 10 else email['date']
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No emails analyzed yet. Run a scan to see results.")

if __name__ == "__main__":
    main()