from models.user import User
from configs import db
from sqlalchemy.exc import NoResultFound, DatabaseError
from typing import Optional

class UserRepository:

    def getByUsernameAndEmail(self, username, email):

        user: Optional[User] = None
        status_code = 200
        try:
            user = db.session.execute(
                db.select(User).filter_by(username=username, email=email)
            ).scalar_one()
        except NoResultFound:
            user = None
        finally:
            db.session.close()

        db.session.close()

        return user

    def delete_all(self):
        
        status_code = 200
        try:
            db.session.query(User).delete()
            db.session.commit()
        except DatabaseError:
            status_code = 400
        finally:
            db.session.close()
            
        return status_code
