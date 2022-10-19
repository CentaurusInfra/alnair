package nvmlcollect

import (
	"bytes"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"strings"
)

const ShellToUse = "bash"

func Shellout(command string) (string, error) {
	var errOut error
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd := exec.Command(ShellToUse, "-c", command) //must use "bash -c + commands", not directly use commands
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		errMsg := fmt.Sprintf("error running command %v: %v", cmd, err)
		errOut = errors.New(errMsg)
		return "", errOut
	}
	if len(stderr.String()) != 0 {
		errMsg := fmt.Sprintf("error returned by command %v: %v", cmd, stderr)
		errOut = errors.New(errMsg)
		return "", errOut
	}
	stdout_str := strings.TrimSuffix(stdout.String(), "\n")
	return stdout_str, nil
}

func GetPodName(pid string) (string, error) {
	cmd := "nsenter --target " + pid + " --uts hostname"
	out, err := Shellout(cmd)
	if err != nil {
		log.Printf(err.Error())
		return "", err
	}
	return out, nil
}
