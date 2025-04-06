from flask import request, Response
from functools import wraps
from werkzeug.security import check_password_hash

USERS = {
    "admin": "pbkdf2:sha256:260000$xyz$32768:8:1$YiBp2vNdQLArA1Bd$bb8d1bfd59674a445942c96dac935c5a29c26a94520d94293b4ab3bb623ff8fa517b3db5d27269ac0d15ac51efe92b0c225cd75f28c439e7a3b34c330e01bac0"
}

def check_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username not in USERS or not check_password_hash(USERS[auth.username], auth.password):
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated
