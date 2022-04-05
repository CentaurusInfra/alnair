/*
Copyright (c) 2022 Futurewei Technologies.
Author: Hao Xu (@hxhp)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <unistd.h>

int register_cgroup(const char* cgroup, const char* alnairID)
{
    struct sockaddr_un addr;
    int ret = 0;
    int data_socket, len;
    char buffer[1000];

    data_socket = socket(AF_UNIX, SOCK_STREAM, 0);
    if(data_socket == -1) {
        fprintf(stderr, "socket creation err.\n");
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, "/run/alnair/alnair.sock", sizeof(addr.sun_path) - 1);
    ret = connect(data_socket, (const struct sockaddr *) &addr, sizeof(addr));

    if(ret == -1) {
        fprintf(stderr, "server is down.\n");
        goto exit;
    }

    sprintf(buffer, "%s %s", cgroup, alnairID);
    len = strlen(cgroup) + strlen(alnairID) + 1;
    buffer[len] = '\n';
    ret = write(data_socket, buffer, len + 1);
    if(ret == -1) {
        fprintf(stderr, "send error.\n");
        goto exit;
    }

    ret = read(data_socket, buffer, sizeof(buffer));
    if(ret == -1) {
        fprintf(stderr, "receive error.\n");
        goto exit;
    }

    if(ret < sizeof(buffer)) buffer[ret] = 0;
    else buffer[sizeof(buffer) - 1] = 0;

    if(strcmp(buffer, "ok") != 0) {
        ret = -2;
        fprintf(stderr, "cgroup registration error: %s\n", buffer);
        goto exit;
    }

    ret = 0;
    
exit:
    close(data_socket);
    return ret;
}
