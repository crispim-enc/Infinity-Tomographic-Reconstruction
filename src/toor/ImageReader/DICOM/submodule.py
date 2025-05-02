

class MainAttribute:
    def __init__(self, value=None, tag=None, type_=None, dependencies=None):
        self._value = value
        self._tag = tag
        self._type = type_
        self._dependencies = dependencies

    def value(self, new):
        if new != self._value:
            self._value = new

        return self._value

    def tag(self, new):
        if new != self._tag:
            self._tag = new

        return self._tag

    def type(self, new):
        if new != self._type:
            self._type = new

        return self._type

    def dependencies(self, new):
        if new != self._dependencies:
            self._dependencies = new

        return self._dependencies


class SubAttribute:
    def __init__(self, value=None, tag=None, type_=None, dependencies=None, codevalue="", codingschemedesignator="",
                 codingschemeversion="", codemeaning="",longcodevalue=" ", urncodevalue=""):
        self._value = value
        self._tag = tag
        self._type = type_
        self._dependencies = dependencies
        self._codeValue = codevalue
        self._codingSchemeDesignator = codingschemedesignator
        self._codingSchemeVersion = codingschemeversion
        self._codeMeaning = codemeaning
        self._longCodeValue = longcodevalue
        self._URNCodeValue = urncodevalue

    def value(self, new):
        if new != self._value:
            self._value = new

        return self._value

    def tag(self, new):
        if new != self._tag:
            self._tag = new

        return self._tag

    def type(self, new):
        if new != self._type:
            self._type = new

        return self._type

    def dependencies(self, new):
        if new != self._dependencies:
            self._dependencies = new

        return self._dependencies

    def codeValue(self, new):
        if new != self._codeValue:
            self._codeValue = new

        return self._codeValue

    def codingSchemeDesignator(self, new):
        if new != self._codingSchemeDesignator:
            self._codingSchemeDesignator = new

        return self._dependencies

    def codingSchemeVersion(self, new):
        if new != self._codingSchemeVersion:
            self._codingSchemeVersion = new

        return self._codingSchemeVersion

    def codeMeaning(self, new):
        if new != self._codeMeaning:
            self._codeMeaning = new

        return self._codeMeaning

    def longCodeValue(self, new):
        if new != self._longCodeValue:
            self._longCodeValue = new

        return self._longCodeValue

    def URNCodeValue(self, new):
        if new != self._URNCodeValue:
            self._URNCodeValue = new

        return self._URNCodeValue











