import json


class Dialog(object):
    ACTOR_SYSTEM = 0
    ACTOR_USER = 1

    ACTORS_MAP = {
        'Tourist': ACTOR_USER,
        'Guide': ACTOR_SYSTEM
    }

    def __init__(self, object_id, session_id):
        self.messages = []
        self.states = []  # Each message has one state.
        self.actors = []  # Each message has an actor id associated.
        self.topic_ids = []
        self.topic_bio = []
        self.object_id = object_id
        self.session_id = session_id

    def add_message(self, text, state, actor, topic_id, topic_bio):
        self.messages.append(text)
        self.states.append(state)
        self.actors.append(actor)
        self.topic_ids.append(topic_id)
        self.topic_bio.append(topic_bio)

    def serialize(self):
        return json.dumps(
            {
                'messages': self.messages,
                'states': self.states,
                'topic_ids': self.topic_ids,
                'topic_bio': self.topic_bio,
                'actors': self.actors,
                'object_id': self.object_id,
                'session_id': self.session_id
            },
            indent=4
        )

    @classmethod
    def deserialize(cls, input_data):
        data = json.loads(input_data)

        obj = Dialog(data['object_id'], data['session_id'])
        obj.messages = data['messages']
        obj.states = data['states']
        obj.actors = data['actors']
        obj.topic_ids = data['topic_ids']
        obj.topic_bio = data['topic_bio']

        return obj
