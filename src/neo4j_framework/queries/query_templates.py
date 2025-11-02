class QueryTemplates:
    """
    Predefined query templates.
    """

    CREATE_NODE = "CREATE (n:Node {name: $name}) RETURN n"
    MATCH_NODE = "MATCH (n:Node {name: $name}) RETURN n"

    @classmethod
    def get_template(cls, template_name: str):
        return getattr(cls, template_name, None)
