class ResponseNotReceivedException(AssertionError):
    """Custom exception raised when no response is received after retries."""

    def __init__(self, message: str = "Response not received after maximum retry attempts.") -> None:
        super().__init__(message)


class SSHCopyFileException(Exception):
    """Custom exception raised when no response is received after retries."""

    def __init__(self, message: str = "Files was not copied.") -> None:
        super().__init__(message)


class ReloadException(Exception):
    """Custom exception raised when the command is not successfully executed."""

    def __init__(self, message: str = "Some services may not have reload correctly.") -> None:
        super().__init__(message)


class FileNotFoundException(Exception):
    """Custom exception raised when file does not exist."""

    def __init__(self, message: str = "File does not exist.") -> None:
        super().__init__(message)


class ExpectedEventsMissingError(AssertionError):
    """Custom exception raised when the expected events are missing."""

    def __init__(self, message: str = "Expected events are missing.") -> None:
        super().__init__(message)


class ServicesNotReadyException(AssertionError):
    """Custom exception raised when services are not ready."""

    def __init__(self, message: str = "Not all services are running.") -> None:
        super().__init__(message)


class PageElementNotFoundException(AssertionError):
    """Custom exception raised when page element is not found."""

    def __init__(self, message: str = "Page element not found.") -> None:
        super().__init__(message)


class ProdDataEmissionsTimeoutError(AssertionError):
    """Custom exception thrown when a specified amount of data is not received in a transmission."""

    def __init__(self, message: str, data: str = "", timeout: int = 0) -> None:
        resolved_message = (
            message if message else f"Insufficient data: {len(data)} characters detected after {timeout}s timeout"
        )
        self.message = resolved_message
        self.data = data
        self.timeout = timeout
        super().__init__(resolved_message, data, timeout)

    def __str__(self) -> str:
        return self.message


class ExpectedModulatorDefinitionError(AssertionError):
    """Custom exception raised when the expected key is missing."""

    def __init__(self, message: str = "", modem: str = "") -> None:
        resolved_message = message if message else f"No modem: {modem} definition has been found in Capabilities"
        self.message = resolved_message
        self.modem = modem
        super().__init__(resolved_message, modem)

    def __str__(self) -> str:
        return self.message


class PageNotExistsException(AssertionError):
    """Custom exception raised when page does not exist."""

    def __init__(self, message: str = "", page: str = "") -> None:
        resolved_message = message if message else f"No '{page}' page found"
        self.message = resolved_message
        self.page = page
        super().__init__(resolved_message, page)

    def __str__(self) -> str:
        return self.message


class NoInfoBoxException(AssertionError):
    """Custom exception raised when text does not appear on page."""

    def __init__(self, message: str = "", text: str = "") -> None:
        resolved_message = message if message else f"No '{text}' info appears"
        self.message = resolved_message
        self.text = text
        super().__init__(resolved_message, text)

    def __str__(self) -> str:
        return self.message


class ExportFileException(AssertionError):
    """Custom exception raised when there is no export file."""

    def __init__(self, message: str = "", text: str = "") -> None:
        resolved_message = message if message else f"No '{text}' export file"
        self.message = resolved_message
        self.text = text
        super().__init__(resolved_message, text)

    def __str__(self) -> str:
        return self.message


class NoTextDecodedException(AssertionError):
    """Custom exception raised when there is no export file."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "No text decoded"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class NoTextException(AssertionError):
    """Custom exception raised when there is no expected text."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "No text received"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class SeviceNotStartedException(AssertionError):
    """Custom exception raised when there is no export file."""

    def __init__(self, message: str = "", service: str = "") -> None:
        resolved_message = message if message else f"Service '{service}' not started"
        self.message = resolved_message
        self.service = service
        super().__init__(resolved_message, service)

    def __str__(self) -> str:
        return self.message


class SpectrumException(AssertionError):
    """Custom exception raised when there is no spectrum displayed."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "No spectrum displayed"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class ElementAttachedException(AssertionError):
    """Custom exception raised when the element is (or is not) present in DOM."""

    def __init__(self, message: str = "", attached: bool = True, name: str = "") -> None:
        resolved_message = message if message else f"'{name}': is {'not' if attached else ''} present in DOM"
        self.message = resolved_message
        self.attached = attached
        self.name = name
        super().__init__(resolved_message, attached, name)

    def __str__(self) -> str:
        return self.message


class SSHClientLogsUnavailableException(AssertionError):
    """Custom exception raised when ssh client is unable to return logs."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "No logs retrieved"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class PictureSizeMismatchException(AssertionError):
    """Custom exception raised when pictures size does not match with expected one."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "Pictures size does not match with expected one"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class NoDetectionException(AssertionError):
    """Custom exception raised when there is no signal detection."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "No signal was detected"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class NoAlarmRaisedException(AssertionError):
    """Custom exception raised when there is no alarm raised."""

    def __init__(self, message: str = "") -> None:
        resolved_message = message if message else "No alarm raised"
        self.message = resolved_message
        super().__init__(resolved_message)

    def __str__(self) -> str:
        return self.message


class AttributeValueException(AssertionError):
    """Custom exception raised when an attribute has different value than expected."""

    def __init__(
        self,
        message: str = "",
        attribute: str = "",
        expected_value: str = "",
        current_val: str = "",
    ) -> None:
        resolved_message = (
            message
            if message
            else f"Waiting for attribute '{attribute}' to change to '{expected_value}'. Current value: {current_val}"
        )
        self.message = resolved_message
        self.attribute = attribute
        self.expected_value = expected_value
        self.current_val = current_val
        super().__init__(resolved_message, attribute, expected_value, current_val)

    def __str__(self) -> str:
        return self.message


class DuplicatedElementException(Exception):
    """Custom exception raised when element with same attribute already exists."""

    def __init__(self, attribute: str) -> None:
        self.message = f"An element with attribute {attribute} already exists."

        super().__init__(self.message)
