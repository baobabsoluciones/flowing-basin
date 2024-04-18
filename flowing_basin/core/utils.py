def custom_serializer(obj):

    """Custom JSON serializer function for handling non-serializable objects"""

    if isinstance(obj, set):
        return list(obj)
    else:
        return str(obj)
